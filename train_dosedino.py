"""Train the paper configuration of DoseDINO."""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
import torch
import torch.optim as optim
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import data_loader_lightning_slice
from dino_unetr.dosedino import DoseDINO
from dino_unetr.tuning_utils import get_llrd_params, inject_lora


def body_masked_l1(
    predicted_dose: torch.Tensor,
    reference_dose: torch.Tensor,
    body_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error over body-mask voxels only."""
    mask = (body_mask > 0.5).to(dtype=predicted_dose.dtype)
    absolute_error = (predicted_dose - reference_dose).abs() * mask
    return absolute_error.sum() / mask.sum().clamp_min(1.0)


class DoseDINOLightningModule(pl.LightningModule):
    """Lightning wrapper for the reported DoseDINO training setup."""

    def __init__(self, cfig: dict, strategy: str = "default"):
        super().__init__()
        self.cfig = cfig
        self.strategy = cfig.get("strategy", strategy)
        model_config = cfig.get("model_params", {})
        self.model = DoseDINO(
            checkpoint_path=model_config.get("checkpoint_path", "dino_unetr/model.pth"),
            input_channels=model_config.get("input_channels", 6),
            output_channels=model_config.get("out_channels", 1),
            rk4_steps=model_config.get("rk4_steps", 4),
        )
        self.lr = float(cfig.get("lr", 1e-4))
        self.weight_decay = float(cfig.get("weight_decay", 1e-4))
        self.num_epochs = int(cfig.get("num_epochs", 200))
        self.scale_out = float(cfig.get("scale_out", 7.5))
        self.scale_loss = float(cfig.get("scale_loss", 4.0))
        self._configure_trainable_parameters()

    def _configure_trainable_parameters(self) -> None:
        if self.strategy == "lora":
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            inject_lora(
                self.model,
                rank=int(self.cfig.get("lora_rank", 8)),
                alpha=int(self.cfig.get("lora_alpha", 8)),
            )
            for name, parameter in self.model.named_parameters():
                if any(
                    component in name
                    for component in ("lora_", "decoder", "head", "geo_encoder", "fusion_layer")
                ):
                    parameter.requires_grad = True
        elif self.strategy == "frozen":
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad = "backbone" not in name
        else:
            for parameter in self.model.parameters():
                parameter.requires_grad = True

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        predicted, _, _ = self.model(batch["data"])
        predicted_dose = predicted * self.scale_out
        loss = body_masked_l1(predicted_dose, batch["label"], batch["body"])
        loss = loss * self.scale_loss
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch["data"].shape[0],
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        if self.strategy == "llrd":
            parameters = get_llrd_params(
                self.model,
                base_lr=self.lr,
                weight_decay=self.weight_decay,
                decay_rate=0.9,
            )
            optimizer = optim.AdamW(parameters, lr=self.lr)
        else:
            parameters = (parameter for parameter in self.model.parameters() if parameter.requires_grad)
            optimizer = optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DoseDINO with PyTorch Lightning.")
    parser.add_argument("cfig_path", nargs="?", default="config_files/config_train.yaml")
    parser.add_argument("--ckpt_path", default=None, help="Lightning checkpoint used to resume training")
    args = parser.parse_args()

    with open(args.cfig_path, encoding="utf-8") as config_file:
        cfig = yaml.safe_load(config_file)

    loaders = data_loader_lightning_slice.GetLoader(cfig=cfig["loader_params"])
    train_loader = loaders.train_dataloader()
    # Validation uses the 148-plan development subset, never the held-out test set.
    val_loader = loaders.val_dataloader()
    model = DoseDINOLightningModule(cfig)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    multi_gpu = torch.cuda.device_count() > 1
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfig["save_model_root"],
        filename="dosedino-best-val-{epoch:03d}-{val_loss:.4f}",
        save_top_k=1,
        save_last=False,
        monitor="val_loss",
        mode="min",
    )
    logger = TensorBoardLogger(save_dir=cfig["save_model_root"], name="dosedino")
    trainer = pl.Trainer(
        max_epochs=int(cfig.get("num_epochs", 200)),
        devices="auto",
        accelerator=accelerator,
        strategy="ddp_find_unused_parameters_true" if multi_gpu else "auto",
        sync_batchnorm=multi_gpu,
        use_distributed_sampler=multi_gpu,
        logger=logger,
        default_root_dir=cfig["save_model_root"],
        callbacks=[LearningRateMonitor(logging_interval="epoch"), checkpoint_callback],
        precision=cfig.get("precision", "16-mixed" if torch.cuda.is_available() else "32-true"),
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)


# Previous public name retained for loading development checkpoints.
GDPDistanceLightningModel = DoseDINOLightningModule


if __name__ == "__main__":
    main()
