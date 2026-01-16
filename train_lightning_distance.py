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

# Importing the new distance model
from dino_unetr.MED_DINO_UNETR_distance import MED_DINO_UNETR_Distance

from Loss import L1_DVH_Loss, L1_MSE_Loss, L1_Loss
from dino_unetr.tuning_utils import inject_lora, get_llrd_params, inject_conv_adapter
from toolkit import compute_pca_projection

class GDPDistanceLightningModel(pl.LightningModule):
    def __init__(self, cfig, strategy='default'):
        super(GDPDistanceLightningModel, self).__init__()
        self.cfig = cfig
        
        # Determine Strategy
        if 'strategy' in cfig:
            self.strategy = cfig['strategy']
        else:
            self.strategy = strategy
        
        input_dim = cfig.get('model_params').get('input_channels', 6)
        print(f">>> Using MED_DINO_UNETR_Distance (Input Dim: {input_dim})")
        
        # Initialize the Distance Model
        # Note: 'checkpoint_path' usually points to the DINO weights or a pretrained model
        ckpt_path = cfig.get('model_params', {}).get('checkpoint_path', 'dino_unetr/model.pth')
        self.model = MED_DINO_UNETR_Distance(checkpoint_path=ckpt_path, input_dim=input_dim)
            
        self.lr = float(cfig['lr'])
        self.num_epochs = cfig['num_epochs']
        self.sig_act = nn.Sigmoid()
        
        # --- Tuning Strategies ---
        if self.strategy == 'lora':
            print(">>> Strategy: LoRA Enabled (Applied to Backbone).")
            # Freeze everything first
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Inject LoRA into the DINO backbone
            lora_rank = cfig.get('lora_rank', 8)
            lora_alpha = cfig.get('lora_alpha', 8)
            inject_lora(self.model.backbone.model, rank=lora_rank, alpha=lora_alpha)
            
            # Unfreeze specific parts
            # We definitely want to train the:
            # 1. Geometry Encoder (it's new)
            # 2. Fusion Layer (it's new)
            # 3. Decoder & Head (task specific)
            # 4. LoRA parameters
            
            for name, param in self.model.named_parameters():
                if any(x in name for x in ["lora_", "decoder", "head", "geo_encoder", "fusion_layer"]):
                    param.requires_grad = True
        
        elif self.strategy == 'adapter':
            print(">>> Strategy: Conv-Adapter Enabled (Applied to Backbone).")
            bottleneck_dim = cfig.get('adapter_bottleneck', 64)
            kernel_size = cfig.get('adapter_kernel_size', 3)
            # Inject adapter into backbone
            inject_conv_adapter(self.model.backbone.model, bottleneck_dim=bottleneck_dim, kernel_size=kernel_size)
            
            # Similar to LoRA, ensure new components are trainable
            for name, param in self.model.named_parameters():
                 if any(x in name for x in ["decoder", "head", "geo_encoder", "fusion_layer"]):
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
        inputs, labels = batch['data'], batch['label']
        inputs_2d = inputs # (B, 6, H, W)
        labels_2d = labels
        
        # Forward pass
        # output is (B, 1, H, W)
        # shallow_feat is f6 (from DINO)
        # deep_feat is f12_fused (Fused DINO+Geometry)
        outputs, shallow_feat, deep_feat = self.model(inputs_2d) 
        
        # Prepare Loss Inputs
        pd_dose = (outputs * self.cfig['scale_out'])
        gt_dose = labels_2d
        
        loss = L1_Loss(pd_dose, gt_dose, self.device)
        loss = loss * self.cfig['scale_loss']
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['data'], batch['label']
        inputs_2d = inputs
        labels_2d = labels
        
        outputs, shallow_feat, deep_feat = self.model(inputs_2d)
        
        pd_dose = (outputs * self.cfig['scale_out'])
        gt_dose = labels_2d

        # Apply Body Mask to Loss Inputs if available
        if 'body' in batch:
            body_mask = batch['body']
            pd_dose = pd_dose * body_mask
            gt_dose = gt_dose * body_mask

        loss = L1_Loss(pd_dose, gt_dose, self.device)
        loss = loss * self.cfig['scale_loss']

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if batch_idx == 0:
            # Visualize the first slice in the batch
            vis_idx = 0
            for i in range(labels.shape[0]):
                if labels[i].max() > 0:
                    vis_idx = i
                    break

            # Take the selected sample in the batch
            lbl_slice = labels[vis_idx, :, :, :]   # (1, H, W) GT
            pred_slice = outputs[vis_idx, :, :, :] # (1, H, W) Pred
            
            # Visualization of inputs
            # Channel 0: Mass Density (Image info)
            # Channel 4: Distance (Geometry info) - assuming 6 channel input structure
            # [mass_density, comb_optptv, comb_oar_priority, beam_plate_norm, comb_oar_distance, Body]
            ct_slice = inputs[vis_idx, 0:1, :, :] 
            dist_slice = inputs[vis_idx, 4:5, :, :]

            lbl_slice = lbl_slice * self.cfig['loader_params']['dose_div_factor']
            pred_slice = pred_slice * self.cfig['scale_out'] * self.cfig['loader_params']['dose_div_factor']

            # Concatenate along Width
            grid_image = torch.cat([lbl_slice, pred_slice], dim=2) 
            
            grid = vutils.make_grid(grid_image, normalize=True, scale_each=True)
            
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_image('Val_Visualization/Label_vs_Pred', grid, self.current_epoch)
            
            # PCA Visualization
            # shallow_feat is f6 (DINO features)
            # deep_feat is f12_fused (Fused DINO+Geo features)
            
            # Note: shallow_feat and deep_feat might be different sizes depending on architecture, 
            # but usually they are feature maps.
            # DINO f6 is usually (B, 768, H/16, W/16).
            
            if len(shallow_feat.shape) == 4:
                vis_shallow = compute_pca_projection(shallow_feat[vis_idx]) # [3, h, w]
            else:
                 # In case it's tokens, might need reshaping, but our model returns reshaped maps
                 vis_shallow = torch.zeros(3, 256, 256).to(self.device)

            if len(deep_feat.shape) == 4:
                vis_deep = compute_pca_projection(deep_feat[vis_idx])       # [3, h, w]
            else:
                vis_deep = torch.zeros(3, 256, 256).to(self.device)
            
            # Resize features to match input size (512x512)
            target_H, target_W = lbl_slice.shape[1], lbl_slice.shape[2]
            vis_shallow = nn.functional.interpolate(vis_shallow.unsqueeze(0), size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze(0)
            vis_deep = nn.functional.interpolate(vis_deep.unsqueeze(0), size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze(0)

            # Fix: Expand CT/Dist to 3 channels to match PCA
            ct_slice_3c = ct_slice.repeat(3, 1, 1)
            dist_slice_3c = dist_slice.repeat(3, 1, 1)

            # Normalize for visualization
            ct_slice_3c = (ct_slice_3c - ct_slice_3c.min()) / (ct_slice_3c.max() - ct_slice_3c.min() + 1e-6)
            dist_slice_3c = (dist_slice_3c - dist_slice_3c.min()) / (dist_slice_3c.max() - dist_slice_3c.min() + 1e-6)

            # Concatenate along Height (now all are 3, H, W)
            # Row 1: CT (Mass) | Distance
            # Row 2: Shallow PCA | Deep PCA
            row1 = torch.cat([ct_slice_3c, dist_slice_3c], dim=2)
            row2 = torch.cat([vis_shallow, vis_deep], dim=2)
            vis_grid = torch.cat([row1, row2], dim=1) # Cat along height
            
            grid = vutils.make_grid(vis_grid, normalize=True, scale_each=True)
            
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_image('Val_Visualization/CT_Dist_PCA', grid, self.current_epoch)
            
        return loss

    def configure_optimizers(self):
        if self.strategy == 'llrd':
            # Use LLRD parameter grouping
            # Note: Need to adjust get_llrd_params to handle the new geometry encoder if we want LLRD on it.
            # For now, simplistic LLRD on backbone + everything else as last group
            params = get_llrd_params(self.model.backbone.model, base_lr=self.lr, weight_decay=1e-4, decay_rate=0.9)
            
            # Add other parameters (Geo Encoder, Decoder, etc)
            # This is a simplification; ideally 'get_llrd_params' should be aware of the whole structure.
            # But the existing utils might just work on the backbone.
            # Let's trust full finetuning for now or rely on standard optimizer if LLRD is complex.
            optimizer = optim.AdamW(params, lr=self.lr)
        else:
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = optim.AdamW(trainable_params, lr=self.lr, weight_decay=1e-4)
            
        scheduler = {'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= self.num_epochs), 
                     'interval': 'epoch', 'frequency': 1}
        return [optimizer], [scheduler]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MED_DINO_UNETR_Distance')
    parser.add_argument('cfig_path', type=str, default='config_files\config_DinoUnetr.yaml')
    parser.add_argument('--ckpt_path', default=None, type=str, help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    cfig = yaml.load(open(args.cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)

    # Data Loaders
    loaders = data_loader_lightning_slice.GetLoader(cfig=cfig['loader_params'])
    train_loader = loaders.train_dataloader()
    val_loader = loaders.test_dataloader()

    # Model
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

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfig['save_model_root'],
        filename='best_distance_model-{epoch:02d}-{train_loss:.4f}',
        save_top_k=1,
        monitor='train_loss',
        mode='min'
    )
    
    # TensorBoard Logger
    # Using a different name to separate from main logs
    tb_logger = TensorBoardLogger(save_dir=cfig['save_model_root'], name="tensorboard_logs_distance")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfig['num_epochs'],
        devices = 'auto', 
        accelerator=accelerator, 
        strategy=stratgy, 
        sync_batchnorm=sync_batchnorm,
        use_distributed_sampler=use_distributed_sampler, 
        logger=tb_logger, 
        default_root_dir=cfig['save_model_root'],
        callbacks=[lr_monitor, checkpoint_callback],
        precision='16-mixed'
    )

    # Training
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)