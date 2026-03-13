"""PyTorch Lightning training entrypoint for distance-aware dose models."""

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import yaml
import argparse
import os
import time
import data_loader_lightning_slice
import copy
from dino_unetr.MED_DINO_UNETR_distance import MED_DINO_UNETR_Distance
from dino_unetr.MED_DINO_UNETR_distance_nmODE import MED_DINO_UNETR_Distance_nmODE
from Loss import L1_DVH_Loss, L1_MSE_Loss, L1_Loss
from dino_unetr.tuning_utils import inject_lora, get_llrd_params
from toolkit import compute_pca_projection

class GDPDistanceLightningModel(pl.LightningModule):
    """LightningModule wrapper for distance-aware MED-DINO-UNETR training."""

    def __init__(self, cfig, strategy='default'):
        """Initialize model, tuning strategy, and optimizer settings.

        Args:
            cfig: Parsed YAML config dictionary.
            strategy: Optional override for fine-tuning strategy.
        """
        super(GDPDistanceLightningModel, self).__init__()
        self.cfig = cfig

        if 'strategy' in cfig:
            self.strategy = cfig['strategy']
        else:
            self.strategy = strategy

        input_dim = cfig.get('model_params').get('input_channels', 6)
        print(f">>> Using MED_DINO_UNETR_Distance (Input Dim: {input_dim})")

        ckpt_path = cfig.get('model_params', {}).get('checkpoint_path', 'dino_unetr/model.pth')
        self.model = MED_DINO_UNETR_Distance(checkpoint_path=ckpt_path, input_dim=input_dim)

        self.lr = float(cfig['lr'])
        self.num_epochs = cfig['num_epochs']
        self.sig_act = nn.Sigmoid()

        self.use_nmODE = cfig.get('use_nmODE', False)

        if self.use_nmODE:
            print(f">>> Using MED_DINO_UNETR_distance_nmODE (Input Dim: {input_dim})")
            self.model = MED_DINO_UNETR_Distance_nmODE(checkpoint_path='dino_unetr\model.pth', input_dim=input_dim)

        if self.strategy == 'lora':
            print(">>> Strategy: LoRA Enabled (Applied to Backbone).")
            for param in self.model.parameters():
                param.requires_grad = False

            lora_rank = cfig.get('lora_rank', 8)
            lora_alpha = cfig.get('lora_alpha', 8)
            inject_lora(self.model, rank=lora_rank, alpha=lora_alpha)

            for name, param in self.model.named_parameters():
                if any(x in name for x in ["lora_", "decoder", "head", "geo_encoder", "fusion_layer"]):
                    param.requires_grad = True

        elif self.strategy == 'llrd':
            print(">>> Strategy: LLRD Enabled.")
            for param in self.model.parameters():
                param.requires_grad = True

        elif self.strategy == 'frozen':
            print(">>> Strategy: Frozen Backbone.")
            for param in self.model.parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                # Unfreeze everything EXCEPT the backbone
                if "backbone" not in name:
                    param.requires_grad = True

        else:
            print(">>> Strategy: Default (Full Finetuning).")
            for param in self.model.parameters():
                param.requires_grad = True

    def training_step(self, batch, batch_idx):
        """Run a single training step on distance-aware inputs."""
        inputs, labels = batch['data'], batch['label']
        inputs_2d = inputs
        labels_2d = labels

        outputs, shallow_feat, deep_feat = self.model(inputs_2d)

        pd_dose = (outputs * self.cfig['scale_out'])
        gt_dose = labels_2d

        loss = L1_Loss(pd_dose, gt_dose, self.device)
        loss = loss * self.cfig['scale_loss']

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Run a single validation step and log visualizations for the first batch."""
        inputs, labels = batch['data'], batch['label']
        inputs_2d = inputs
        labels_2d = labels

        outputs, shallow_feat, deep_feat = self.model(inputs_2d)

        pd_dose = (outputs * self.cfig['scale_out'])
        gt_dose = labels_2d

        if 'body' in batch:
            body_mask = batch['body']
            pd_dose = pd_dose * body_mask
            gt_dose = gt_dose * body_mask

        loss = L1_Loss(pd_dose, gt_dose, self.device)
        loss = loss * self.cfig['scale_loss']

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            # Visualize the first non-empty slice
            vis_idx = 0
            for i in range(labels.shape[0]):
                if labels[i].max() > 0:
                    vis_idx = i
                    break

            lbl_slice = labels[vis_idx, :, :, :]
            pred_slice = outputs[vis_idx, :, :, :]

            # [mass_density, comb_optptv, comb_oar_priority, beam_plate_norm, comb_oar_distance, Body]
            ct_slice = inputs[vis_idx, 0:1, :, :]
            dist_slice = inputs[vis_idx, 4:5, :, :]

            lbl_slice = lbl_slice * self.cfig['loader_params']['dose_div_factor']
            pred_slice = pred_slice * self.cfig['scale_out'] * self.cfig['loader_params']['dose_div_factor']

            grid_image = torch.cat([lbl_slice, pred_slice], dim=2)
            grid = vutils.make_grid(grid_image, normalize=True, scale_each=True)

            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_image('Val_Visualization/Label_vs_Pred', grid, self.current_epoch)

            if len(shallow_feat.shape) == 4:
                vis_shallow = compute_pca_projection(shallow_feat[vis_idx])
            else:
                vis_shallow = torch.zeros(3, 256, 256).to(self.device)

            if len(deep_feat.shape) == 4:
                vis_deep = compute_pca_projection(deep_feat[vis_idx])
            else:
                vis_deep = torch.zeros(3, 256, 256).to(self.device)

            target_H, target_W = lbl_slice.shape[1], lbl_slice.shape[2]
            vis_shallow = nn.functional.interpolate(vis_shallow.unsqueeze(0), size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze(0)
            vis_deep = nn.functional.interpolate(vis_deep.unsqueeze(0), size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze(0)

            ct_slice_3c = ct_slice.repeat(3, 1, 1)
            dist_slice_3c = dist_slice.repeat(3, 1, 1)

            ct_slice_3c = (ct_slice_3c - ct_slice_3c.min()) / (ct_slice_3c.max() - ct_slice_3c.min() + 1e-6)
            dist_slice_3c = (dist_slice_3c - dist_slice_3c.min()) / (dist_slice_3c.max() - dist_slice_3c.min() + 1e-6)

            # Row 1: CT | Distance, Row 2: PCA features
            row1 = torch.cat([ct_slice_3c, dist_slice_3c], dim=2)
            row2 = torch.cat([vis_shallow, vis_deep], dim=2)
            vis_grid = torch.cat([row1, row2], dim=1)

            grid = vutils.make_grid(vis_grid, normalize=True, scale_each=True)

            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_image('Val_Visualization/CT_Dist_PCA', grid, self.current_epoch)

        return loss

    def configure_optimizers(self):
        """Create optimizer and scheduler based on the tuning strategy."""
        if self.strategy == 'llrd':
            # Use LLRD parameter grouping
            params = get_llrd_params(self.model, base_lr=self.lr, weight_decay=1e-4, decay_rate=0.9)
            optimizer = optim.AdamW(params, lr=self.lr)
        else:
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = optim.AdamW(trainable_params, lr=self.lr, weight_decay=1e-4)

        scheduler = {
            'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MED-DINO-UNETR-Distance models with Lightning.')
    parser.add_argument('cfig_path', type=str, nargs='?', default='config_files\config_train.yaml')
    parser.add_argument('--ckpt_path', default=None, type=str, help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    cfig = yaml.load(open(args.cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)

    loaders = data_loader_lightning_slice.GetLoader(cfig=cfig['loader_params'])
    train_loader = loaders.train_dataloader()
    val_loader = loaders.test_dataloader()

    model = GDPDistanceLightningModel(cfig)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
        stratgy = 'ddp_find_unused_parameters_true'
        sync_batchnorm = True
        use_distributed_sampler = True
    else:
        stratgy = 'auto'
        sync_batchnorm = False
        use_distributed_sampler = False

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback_train = ModelCheckpoint(
        dirpath=cfig['save_model_root'],
        filename='dinounetr-best-train-{epoch:02d}-{train_loss:.4f}',
        save_top_k=1,
        monitor='train_loss',
        mode='min'
    )

    checkpoint_callback_val = ModelCheckpoint(
        dirpath=cfig['save_model_root'],
        filename='dinounetr-best-val-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    tb_logger = TensorBoardLogger(save_dir=cfig['save_model_root'], name="tensorboard_logs_distance")

    trainer = pl.Trainer(
        max_epochs=cfig['num_epochs'],
        devices='auto',
        accelerator=accelerator,
        strategy=stratgy,
        sync_batchnorm=sync_batchnorm,
        use_distributed_sampler=use_distributed_sampler,
        logger=tb_logger,
        default_root_dir=cfig['save_model_root'],
        callbacks=[lr_monitor, checkpoint_callback_train, checkpoint_callback_val],
        precision='16-mixed'
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)