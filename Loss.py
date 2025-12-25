import torch.nn as nn
import torch


class DVHLoss(nn.Module):
    def __init__(self):
        super(DVHLoss, self).__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, predicted_dose, mask, dose):
        """
        Calculate DVH loss: averaged over all OARs. 
        Supports ONLY 2D slices (N, C, H, W).
        """
        
        # Spatial dimensions for 4D input (N, C, H, W)
        spatial_dims = (2, 3)

        # Calculate predicted hist
        vols = torch.sum(mask, axis=spatial_dims) # (N, 3)
        vols = vols.clamp(min=1) # Avoid division by zero


        momentOfStructure = {'PTV': {'moments': [2, 4, 6], 'weights': [1, 1, 1]},
                'serial_OAR': {'moments': [5, 10], 'weights': [1, 1]},
                'parallel_OAR': {'moments': [1, 2], 'weights': [1, 1]}}

        keys = list(momentOfStructure.keys())
        #momentOfStructure = dict([(k, v) for k in keys for v in values])
        oarPredMoment = []
        oarRealMoment = []
        pres = 60
        epsilon = 0.00001#Added epsilon as the loss function can become sqrt(0)


        for i in range(mask.shape[1]):
            moments = momentOfStructure[keys[i]]['moments']
            weights = momentOfStructure[keys[i]]['weights']
            
            # 提取当前器官的 mask (N, H, W)
            current_mask = mask[:, i, :, :] 
                
            current_vol = vols[:, i]
            
            # 预计算 masked dose
            oarpreddose_masked = predicted_dose * current_mask
            oarRealDose_masked = dose * current_mask

            for j in range(len(moments)):
                gEUDa = moments[j]
                weight = weights[j]

                term_pred = torch.sum(torch.pow(oarpreddose_masked, gEUDa), axis=spatial_dims)
                term_real = torch.sum(torch.pow(oarRealDose_masked, gEUDa), axis=spatial_dims)
                
                oarPredMomenta = weight * torch.pow((1 / current_vol) * term_pred + epsilon, 1 / gEUDa)
                oarRealMomenta = weight * torch.pow((1 / current_vol) * term_real + epsilon, 1 / gEUDa)

                oarPredMoment.append(oarPredMomenta)
                oarRealMoment.append(oarRealMomenta)
                
        oarPredMoment = torch.stack(oarPredMoment)
        oarRealMoment = torch.stack(oarRealMoment)

        return self.loss(oarPredMoment, oarRealMoment)
    
def L1_DVH_Loss(pd_dose, gt_dose,ptv_mask,oar_serial_mask,oar_parallel_mask,device,weight=0.01):
    dvh_loss_fn = DVHLoss() 
    
    roi = torch.cat((ptv_mask,oar_serial_mask,oar_parallel_mask), dim=1).type(torch.bool)

    dvhloss = dvh_loss_fn(pd_dose,roi,gt_dose)

    L1Loss = nn.L1Loss().to(device)
    mae_loss = L1Loss(pd_dose,gt_dose)

    total_loss = weight*dvhloss + mae_loss


    return total_loss, dvhloss, mae_loss

def L1_Loss(pd_dose, gt_dose, device):
    L1Loss = nn.L1Loss().to(device)
    mae_loss = L1Loss(pd_dose,gt_dose)
    return mae_loss

def L1_MSE_Loss(pd_dose, gt_dose, device, mse_weight=0.1):
    """
    Combined L1 and MSE Loss.
    """
    l1_loss_fn = nn.L1Loss().to(device)
    mse_loss_fn = nn.MSELoss().to(device)
    
    l1_loss = l1_loss_fn(pd_dose, gt_dose)
    mse_loss = mse_loss_fn(pd_dose, gt_dose)
    
    total_loss = l1_loss + mse_weight * mse_loss
    
    return total_loss, l1_loss, mse_loss
