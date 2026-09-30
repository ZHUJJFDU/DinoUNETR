"""Compatibility entrypoint for the former distance-model training script."""

from train_dosedino import (
    DoseDINOLightningModule,
    body_masked_l1,
    main,
)

GDPDistanceLightningModel = DoseDINOLightningModule

__all__ = ["DoseDINOLightningModule", "GDPDistanceLightningModel", "body_masked_l1"]


if __name__ == "__main__":
    main()
