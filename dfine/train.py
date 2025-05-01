import os
import json
import torch
import einops
import numpy as np
from pathlib import Path
from datetime import datetime
from .control_utils import compute_gramians
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard.writer import SummaryWriter
from .memory import ReplayBuffer
from .configs import TrainConfig
from .models import (
    Dfine,
    YDecoder,
    ZDecoder,
    Encoder,
)


def train_dfine(
    config: TrainConfig,
    train_replay_buffer: ReplayBuffer,
    test_replay_buffer: ReplayBuffer,
):

    # prepare logging
    log_dir = Path(config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(log_dir, exist_ok=True)
    with open(log_dir / "args.json", "w") as f:
        json.dump(config.dict(), f)
    
    writer = SummaryWriter(log_dir=log_dir)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # define models and optimizer
    device = config.device

    encoder = Encoder(
        y_dim=train_replay_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
    ).to(device)

    y_decoder = YDecoder(
        y_dim=train_replay_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
    ).to(device)

    dfine = Dfine(
        x_dim=config.x_dim,
        u_dim=train_replay_buffer.u_dim,
        a_dim=config.a_dim,
        device=device,
    ).to(device)

    all_params = (
        list(encoder.parameters()) +
        list(y_decoder.parameters()) + 
        list(dfine.parameters())
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # train and test loop
    for update in range(config.num_updates):

        # train
        encoder.train()
        dfine.train()
        y_decoder.train()

        y, u, _, _ = train_replay_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")

        # initial belief over x0: N(0, I)
        mean = torch.zeros((config.batch_size, config.x_dim), device=device)
        cov = torch.eye(config.x_dim, device=device).repeat([config.batch_size, 1, 1])
        
        y_pred_loss = 0
        
        for t in range(config.chunk_length - config.prediction_k - 1):
            mean, cov = dfine.dynamics_update(
                mean=mean,
                cov=cov,
                u=u[t],
            )
            mean, cov = dfine.measurement_update(
                mean=mean,
                cov=cov,
                a=a[t+1],
            )

            # tensors to hold predictions of future ys & cs & as
            pred_y = torch.zeros((config.prediction_k, config.batch_size, train_replay_buffer.y_dim), device=device)

            pred_mean = mean
            pred_cov = cov

            for k in range(config.prediction_k):
                pred_mean, pred_cov = dfine.dynamics_update(
                    mean=pred_mean,
                    cov=pred_cov,
                    u=u[t+k+1]
                )
                pred_y[k] = y_decoder(pred_mean @ dfine.C.T)

            true_y = y[t+2:t+2+config.prediction_k]
            true_y_flatten = einops.rearrange(true_y, "k b y -> (k b) y")
            pred_y_flatten = einops.rearrange(pred_y, "k b y -> (k b) y")
            y_pred_loss += criterion(pred_y_flatten, true_y_flatten)

        y_pred_loss /= (config.chunk_length - config.prediction_k - 1)

        # balancing loss
        Wc, Wo = compute_gramians(
            A=dfine.A,
            B=dfine.B,
            C=dfine.C
        )
        balancing_loss = 1 / torch.trace(Wc @ Wo)

        total_loss = y_pred_loss + config.balancing_weight * balancing_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()

        writer.add_scalar("train/ y prediction loss", y_pred_loss.item(), update)
        writer.add_scalar("train/ balancing loss", balancing_loss.item(), update)
        print(f"update step: {update+1}, train_loss: {total_loss.item()}")

        # test
        if update % config.test_interval == 0:
            # test
            encoder.eval()
            dfine.eval()
            y_decoder.eval()

            y, u, _, _ = test_replay_buffer.sample(
                batch_size=config.batch_size,
                chunk_length=config.chunk_length,
            )

            # convert to tensor, transform to device, reshape to time-first
            y = torch.as_tensor(y, device=device)
            y = einops.rearrange(y, "b l y -> l b y")
            a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
            a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
            u = torch.as_tensor(u, device=device)
            u = einops.rearrange(u, "b l u -> l b u")

            # initial belief over x0: N(0, I)
            mean = torch.zeros((config.batch_size, config.x_dim), device=device)
            cov = torch.eye(config.x_dim, device=device).repeat([config.batch_size, 1, 1])
            
            y_pred_loss = 0
            
            for t in range(config.chunk_length - config.prediction_k - 1):
                mean, cov = dfine.dynamics_update(
                    mean=mean,
                    cov=cov,
                    u=u[t],
                )
                mean, cov = dfine.measurement_update(
                    mean=mean,
                    cov=cov,
                    a=a[t+1],
                )

                # tensors to hold predictions of future ys & cs & as
                pred_y = torch.zeros((config.prediction_k, config.batch_size, train_replay_buffer.y_dim), device=device)

                pred_mean = mean
                pred_cov = cov

                for k in range(config.prediction_k):
                    pred_mean, pred_cov = dfine.dynamics_update(
                        mean=pred_mean,
                        cov=pred_cov,
                        u=u[t+k+1]
                    )
                    pred_y[k] = y_decoder(pred_mean @ dfine.C.T)

                true_y = y[t+2:t+2+config.prediction_k]
                true_y_flatten = einops.rearrange(true_y, "k b y -> (k b) y")
                pred_y_flatten = einops.rearrange(pred_y, "k b y -> (k b) y")
                y_pred_loss += criterion(pred_y_flatten, true_y_flatten)

            y_pred_loss /= (config.chunk_length - config.prediction_k - 1)

            # balancing loss
            Wc, Wo = compute_gramians(
                A=dfine.A,
                B=dfine.B,
                C=dfine.C
            )
            balancing_loss = 1 / torch.trace(Wc @ Wo)

            total_loss = y_pred_loss + config.balancing_weight * balancing_loss
            
            writer.add_scalar("test/ y prediction loss", y_pred_loss.item(), update)
            writer.add_scalar("test/ balancing loss", balancing_loss.item(), update)
            print(f"evaluation step: {update+1}, test_loss: {total_loss.item()}")

    torch.save(encoder.state_dict(), log_dir / "encoder.pth")
    torch.save(y_decoder.state_dict(), log_dir / "y_decoder.pth")
    torch.save(dfine.state_dict(), log_dir / "dfine.pth")

    return {"model_dir": log_dir}


def train_z_decoder(
    train_replay_buffer: ReplayBuffer,
    test_replay_buffer: ReplayBuffer,
    backbone_dir: Path,
):

    with open(backbone_dir / "args.json", "r") as f:
        config = TrainConfig(**json.load(f))

    # prepare logging
    log_dir = Path(config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=log_dir)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # define models and optimizer
    device = config.device

    encoder = Encoder(
        y_dim=train_replay_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
    ).to(device)

    y_decoder = YDecoder(
        y_dim=train_replay_buffer.y_dim,
        a_dim=config.a_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
    ).to(device)

    dfine = Dfine(
        x_dim=config.x_dim,
        u_dim=train_replay_buffer.u_dim,
        a_dim=config.a_dim,
        device=device,
    ).to(device)

    z_decoder = ZDecoder(
        x_dim=config.x_dim,
        z_dim=train_replay_buffer.z_dim,
        hidden_dim=config.hidden_dim,
        dropout_p=config.dropout_p,
    ).to(device)

    # load the backbone
    encoder.load_state_dict(torch.load(backbone_dir / "encoder.pth", weights_only=True))
    dfine.load_state_dict(torch.load(backbone_dir / "dfine.pth", weights_only=True))
    y_decoder.load_state_dict(torch.load(backbone_dir / "y_decoder.pth", weights_only=True))

    # freeze backbone models
    for p in encoder.parameters():
        p.requires_grad = False

    for p in y_decoder.parameters():
        p.requires_grad = False

    for p in dfine.parameters():
        p.requires_grad = False

    encoder.eval()
    dfine.eval()
    y_decoder.eval()

    all_params = (
        list(z_decoder.parameters())       
    )

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(all_params, lr=config.lr, eps=config.eps)

    # train and test loop
    for update in range(config.num_updates):
        # train
        z_decoder.train()

        y, u, z, _ = train_replay_buffer.sample(
            batch_size=config.batch_size,
            chunk_length=config.chunk_length,
        )

        # convert to tensor, transform to device, reshape to time-first
        y = torch.as_tensor(y, device=device)
        y = einops.rearrange(y, "b l y -> l b y")
        a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
        a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
        u = torch.as_tensor(u, device=device)
        u = einops.rearrange(u, "b l u -> l b u")
        z = torch.as_tensor(z, device=device)
        z = einops.rearrange(z, "b l z -> l b z")

        # initial belief over x0: N(0, I)
        mean = torch.zeros((config.batch_size, config.x_dim), device=device)
        cov = torch.eye(config.x_dim, device=device).repeat([config.batch_size, 1, 1])
        
        recon_z = torch.zeros(
            (config.chunk_length-config.prediction_k-1, config.batch_size, train_replay_buffer.z_dim),
            device=device,
        )
        z_pred_loss = 0

        for t in range(config.chunk_length - config.prediction_k - 1):
            mean, cov = dfine.dynamics_update(
                mean=mean,
                cov=cov,
                u=u[t],
            )
            mean, cov = dfine.measurement_update(
                mean=mean,
                cov=cov,
                a=a[t+1],
            )

            # z reconstruction
            recon_z[t] = z_decoder(mean)

            # tensors to hold predictions of future zs
            pred_z = torch.zeros((config.prediction_k, config.batch_size, train_replay_buffer.z_dim), device=device)

            pred_mean = mean
            pred_cov = cov

            for k in range(config.prediction_k):
                pred_mean, pred_cov = dfine.dynamics_update(
                    mean=pred_mean,
                    cov=pred_cov,
                    u=u[t+k+1]
                )
                pred_z[k] = z_decoder(pred_mean)

            true_z = z[t+2:t+2+config.prediction_k]
            true_z_flatten = einops.rearrange(true_z, "k b z -> (k b) z")
            pred_z_flatten = einops.rearrange(pred_z, "k b z -> (k b) z")
            z_pred_loss += criterion(pred_z_flatten, true_z_flatten)

        z_pred_loss /= (config.chunk_length - config.prediction_k - 1)

        # z reconstruction loss
        z_flatten = einops.rearrange(
            z[1:config.chunk_length - config.prediction_k],
            "L b z -> (L b) z"
        )
        recon_z = einops.rearrange(recon_z, "L b z -> (L b) z")
        z_recon_loss = criterion(recon_z, z_flatten)
        
        optimizer.zero_grad()
        z_recon_loss.backward()
        clip_grad_norm_(all_params, config.clip_grad_norm)
        optimizer.step()

        writer.add_scalar("train/ z reconstruction loss", z_recon_loss.item(), update)
        writer.add_scalar("train/ z prediction loss", z_pred_loss.item(), update)
        print(f"update step: {update+1}, train_loss: {z_recon_loss.item()}")

        # test
        if update % config.test_interval == 0:
 
            z_decoder.eval()

            y, u, z, _ = test_replay_buffer.sample(
                batch_size=config.batch_size,
                chunk_length=config.chunk_length,
            )

            # convert to tensor, transform to device, reshape to time-first
            y = torch.as_tensor(y, device=device)
            y = einops.rearrange(y, "b l y -> l b y")
            a = encoder(einops.rearrange(y, "l b y -> (l b) y"))
            a = einops.rearrange(a, "(l b) a -> l b a", b=config.batch_size)
            u = torch.as_tensor(u, device=device)
            u = einops.rearrange(u, "b l u -> l b u")
            z = torch.as_tensor(z, device=device)
            z = einops.rearrange(z, "b l z -> l b z")

            # initial belief over x0: N(0, I)
            mean = torch.zeros((config.batch_size, config.x_dim), device=device)
            cov = torch.eye(config.x_dim, device=device).repeat([config.batch_size, 1, 1])
            
            recon_z = torch.zeros(
                (config.chunk_length-config.prediction_k-1, config.batch_size, train_replay_buffer.z_dim),
                device=device,
            )
            z_pred_loss = 0

            for t in range(config.chunk_length - config.prediction_k - 1):
                mean, cov = dfine.dynamics_update(
                    mean=mean,
                    cov=cov,
                    u=u[t],
                )
                mean, cov = dfine.measurement_update(
                    mean=mean,
                    cov=cov,
                    a=a[t+1],
                )

                # z reconstruction
                recon_z[t] = z_decoder(mean)

                # tensors to hold predictions of future zs
                pred_z = torch.zeros((config.prediction_k, config.batch_size, train_replay_buffer.z_dim), device=device)

                pred_mean = mean
                pred_cov = cov

                for k in range(config.prediction_k):
                    pred_mean, pred_cov = dfine.dynamics_update(
                        mean=pred_mean,
                        cov=pred_cov,
                        u=u[t+k+1]
                    )
                    pred_z[k] = z_decoder(pred_mean)

                true_z = z[t+2:t+2+config.prediction_k]
                true_z_flatten = einops.rearrange(true_z, "k b z -> (k b) z")
                pred_z_flatten = einops.rearrange(pred_z, "k b z -> (k b) z")
                z_pred_loss += criterion(pred_z_flatten, true_z_flatten)

            z_pred_loss /= (config.chunk_length - config.prediction_k - 1)

            # z reconstruction loss
            z_flatten = einops.rearrange(
                z[1:config.chunk_length - config.prediction_k],
                "L b z -> (L b) z"
            )
            recon_z = einops.rearrange(recon_z, "L b z -> (L b) z")
            z_recon_loss = criterion(recon_z, z_flatten)
            
            writer.add_scalar("test/ z reconstruction loss", z_recon_loss.item(), update)
            writer.add_scalar("test/ z prediction loss", z_pred_loss.item(), update)
            print(f"evaluation step: {update+1}, test_loss: {z_recon_loss.item()}")


    torch.save(z_decoder.state_dict(), log_dir / "z_decoder.pth")

    return {"model_dir": log_dir}