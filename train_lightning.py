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
from dino_unetr.DINO_UNETR import DINO_UNETR
from dino_unetr.MED_DINO_UNETR import MED_DINO_UNETR as MED_DINO_UNETR_Basic
from dino_unetr.MED_DINO_UNETR_layout import MED_DINO_UNETR as MED_DINO_UNETR_Layout
from dino_unetr.MED_DINO_UNETR_nmODE import MED_DINO_UNETR as MED_DINO_UNETR_nmODE
from Loss import L1_DVH_Loss, L1_MSE_Loss, L1_Loss
from dino_unetr.tuning_utils import inject_lora, get_llrd_params, inject_mdt_adapter
from toolkit import compute_pca_projection

class GDPLightningModel(pl.LightningModule):
    def __init__(self, cfig, strategy='default'):
        super(GDPLightningModel, self).__init__()
        self.cfig = cfig
        
        # Determine Strategy
        if 'strategy' in cfig:
            self.strategy = cfig['strategy']
        else:
            self.strategy = strategy
        
        # Determine Model Layout
        self.use_layout = cfig.get('layout', False)
        self.use_nmODE = cfig.get('use_nmODE', False)
        
        input_dim = cfig.get('model_params').get('input_channels')

        if self.use_layout:
            print(f">>> Using MED_DINO_UNETR_Layout (Input Dim: {input_dim})")
            self.model = MED_DINO_UNETR_Layout(checkpoint_path='dino_unetr\model.pth', input_dim=input_dim)
        elif self.use_nmODE:
            print(f">>> Using MED_DINO_UNETR_nmODE (Input Dim: {input_dim})")
            self.model = MED_DINO_UNETR_nmODE(checkpoint_path='dino_unetr\model.pth', input_dim=input_dim)
        else:
            print(f">>> Using MED_DINO_UNETR_Basic (Input Dim: {input_dim})")
            self.model = MED_DINO_UNETR_Basic(checkpoint_path='dino_unetr\model.pth', input_dim=input_dim)
            
        self.lr = float(cfig['lr'])
        self.num_epochs = cfig['num_epochs']
        self.sig_act = nn.Sigmoid()
        
        if self.strategy == 'lora':
            print(">>> Strategy: LoRA Enabled.")
            for param in self.model.parameters():
                param.requires_grad = False
                
            lora_rank = cfig.get('lora_rank', 8)
            lora_alpha = cfig.get('lora_alpha', 8)
            inject_lora(self.model, rank=lora_rank, alpha=lora_alpha)
            
            for name, param in self.model.named_parameters():
                if "lora_" in name or "decoder" in name or "head" in name or "patch_embed" in name:
                    param.requires_grad = True
        

        elif self.strategy == 'mdt':
            print(">>> Strategy: MDT Adapter Enabled.")
            reduction = cfig.get('mdt_reduction', 4)
            inject_mdt_adapter(self.model, reduction=reduction)
            # inject_mdt_adapter also handles freezing and unfreezing logic
                    
        elif self.strategy == 'llrd':
            print(">>> Strategy: LLRD Enabled.")
            for param in self.model.parameters():
                param.requires_grad = True
        
        elif self.strategy == 'frozen':
            print(">>> Strategy: Frozen Backbone.")
            for param in self.model.parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                if "decoder" in name or "head" in name or "patch_embed" in name:
                    param.requires_grad = True
            
        else:
            print(">>> Strategy: Default (Full Finetuning).")
            for param in self.model.parameters():
                param.requires_grad = True

    def training_step(self, batch, batch_idx):
        inputs, labels = batch['data'], batch['label']
        # Input is already 2D slices (B, C, H, W) from SliceDataset
        inputs_2d = inputs
        labels_2d = labels
        
        # # 获取 DVH Loss 所需的掩膜
        # # Loss.py 中的 L1_DVH_Loss 现在期望输入维度为 (N, C, H, W) 
        # ptv_mask = batch['ptv']
        # oar_serial_mask = batch['oar_serial']
        # oar_parallel_mask = batch['oar_parallel']
        
        if self.use_layout:
            layout_data = {
                'spacing': batch['spacing'],
                'isocenter': batch['isocenter'],
                'angle_list': batch['angle_list']
            }
            outputs,_,_ = self.model(inputs_2d, layout_data)
        else:
            outputs,_,_ = self.model(inputs_2d) 
        
        # Prepare Loss Inputs
        pd_dose = (outputs * self.cfig['scale_out'])
        gt_dose = labels_2d
        
        # 计算 Loss
        # loss, dvh_loss, mae_loss = L1_DVH_Loss(
        #     pd_dose, 
        #     gt_dose, 
        #     ptv_mask, 
        #     oar_serial_mask, 
        #     oar_parallel_mask, 
        #     self.device, 
        #     weight=0.01
        # )
        loss = L1_Loss(pd_dose, gt_dose, self.device)
        # loss, l1_loss, mse_loss = L1_MSE_Loss(pd_dose, gt_dose, self.device)
        loss = loss * self.cfig['scale_loss']
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log('train_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log('dvh_loss', dvh_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log('mae_loss', mae_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['data'], batch['label']
        inputs_2d = inputs
        labels_2d = labels
        
        # ptv_mask = batch['ptv']
        # oar_serial_mask = batch['oar_serial']
        # oar_parallel_mask = batch['oar_parallel']

        if self.use_layout:
            layout_data = {
                'spacing': batch['spacing'],
                'isocenter': batch['isocenter'],
                'angle_list': batch['angle_list']
            }
            outputs, shallow_feat, deep_feat = self.model(inputs_2d, layout_data)
        else:
            outputs, shallow_feat, deep_feat = self.model(inputs_2d)
        
        pd_dose = (outputs * self.cfig['scale_out'])
        gt_dose = labels_2d

        # Apply Body Mask to Loss Inputs
        body_mask = batch['body']
        pd_dose = pd_dose * body_mask
        gt_dose = gt_dose * body_mask

        # loss, dvh_loss, mae_loss = L1_DVH_Loss(
        #     pd_dose, 
        #     gt_dose, 
        #     ptv_mask, 
        #     oar_serial_mask, 
        #     oar_parallel_mask, 
        #     self.device, 
        #     weight=0.01
        # )
        loss = L1_Loss(pd_dose, gt_dose, self.device)
        # loss, l1_loss, mse_loss = L1_MSE_Loss(pd_dose, gt_dose, self.device)
        loss = loss * self.cfig['scale_loss']

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_l1_loss', l1_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log('val_mse_loss', mse_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log('val_dvh_loss', dvh_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # self.log('val_mae_loss', mae_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        

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
            ct_slice = inputs[vis_idx, 0:1, :, :] # (1, H, W) CT (mass_density) 

            lbl_slice = lbl_slice * self.cfig['loader_params']['dose_div_factor']
            pred_slice = pred_slice * self.cfig['scale_out'] * self.cfig['loader_params']['dose_div_factor']

            # Concatenate along Width
            grid_image = torch.cat([lbl_slice, pred_slice], dim=2) 
            
            grid = vutils.make_grid(grid_image, normalize=True, scale_each=True)
            
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_image('Val_Visualization/Input_Label_Pred', grid, self.current_epoch)
            
            # PCA Visualization
            vis_shallow = compute_pca_projection(shallow_feat[0]) # [3, h, w]
            vis_deep = compute_pca_projection(deep_feat[0])       # [3, h, w]
            
            # Resize features to match input size (512x512)
            target_H, target_W = lbl_slice.shape[1], lbl_slice.shape[2]
            vis_shallow = nn.functional.interpolate(vis_shallow.unsqueeze(0), size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze(0)
            vis_deep = nn.functional.interpolate(vis_deep.unsqueeze(0), size=(target_H, target_W), mode='bilinear', align_corners=False).squeeze(0)

            # Fix: Expand CT to 3 channels to match PCA (1->3)
            ct_slice_3c = ct_slice.repeat(3, 1, 1)
            # Optional: Normalize CT to 0-1 for better visualization
            ct_slice_3c = (ct_slice_3c - ct_slice_3c.min()) / (ct_slice_3c.max() - ct_slice_3c.min() + 1e-6)

            # Concatenate along Height (now all are 3, H, W)
            vis_grid = torch.cat([ct_slice_3c, vis_shallow, vis_deep], dim=2)
            
            grid = vutils.make_grid(vis_grid, normalize=True, scale_each=True)
            
            if hasattr(self.logger, 'experiment'):
                self.logger.experiment.add_image('Val_Visualization/Input_PCA_Shallow_Deep', grid, self.current_epoch)
            
        return loss

    def configure_optimizers(self):
        if self.strategy == 'llrd':
            # Use LLRD parameter grouping
            params = get_llrd_params(self.model, base_lr=self.lr, weight_decay=1e-4, decay_rate=0.9)
            optimizer = optim.AdamW(params, lr=self.lr)
        else:
            trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = optimizer = optim.AdamW(trainable_params, lr=self.lr, weight_decay=1e-4)
            
        scheduler = {'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= self.num_epochs), 
                     'interval': 'epoch', 'frequency': 1}
        return [optimizer], [scheduler]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('cfig_path', type=str, nargs='?',default='config_files\config_DinoUnetr.yaml')
    parser.add_argument('--ckpt_path', default=None, type=str, help='Path to checkpoint to resume training from')
    args = parser.parse_args()

    cfig = yaml.load(open(args.cfig_path, encoding='utf-8'), Loader=yaml.FullLoader)

    # Standard Training Logic
    # Data Loaders
    loaders = data_loader_lightning_slice.GetLoader(cfig=cfig['loader_params'])
    train_loader = loaders.train_dataloader()
    val_loader = loaders.test_dataloader()

    # Model
    model = GDPLightningModel(cfig) 

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
    
    # TensorBoard Logger
    # Using a different name to separate from main logs
    tb_logger = TensorBoardLogger(save_dir=cfig['save_model_root'], name="tensorboard_logs")

    # Trainer
    max_steps = cfig.get('max_steps', -1)
    max_epochs = cfig['num_epochs'] if max_steps == -1 else -1

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        devices = 'auto', 
        accelerator=accelerator, 
        strategy=stratgy, 
        sync_batchnorm=sync_batchnorm,
        use_distributed_sampler=use_distributed_sampler, 
        logger=tb_logger, 
        default_root_dir=cfig['save_model_root'],
        callbacks=[lr_monitor, checkpoint_callback_train, checkpoint_callback_val],
        precision='16-mixed'
    )

    # Training
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt_path)
    # trainer.fit(model, datamodule=dm)