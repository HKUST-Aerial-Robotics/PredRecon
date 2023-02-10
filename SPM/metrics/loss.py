import torch
import torch.nn as nn
import torch.nn.functional as F

from extensions.chamfer_distance.chamfer_distance import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance


CD = ChamferDistance()
EMD = EarthMoverDistance()


def cd_loss_L1(pcs1, pcs2):
    r"""
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0


def cd_loss_L2(pcs1, pcs2):
    r"""
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    return torch.mean(dist1) + torch.mean(dist2)


def emd_loss(pcs1, pcs2):
    r"""
    EMD Loss.

    Args:
        xyz1 (torch.Tensor): (b, N, 3)
        xyz2 (torch.Tensor): (b, N, 3)
    """
    dists = EMD(pcs1, pcs2)
    return torch.mean(dists)

def scale_est_loss(scale_p, prob_p, scale_gt, prob_gt):
    r'''
    Loss func for Scale Estimator.

    Args:
        scale_p (torch.Tensor): (b, 3)
        prob_p (torch.Tensor): (b, 3, 5)
        scale_gt (torch.Tensor): (b, 3) GT
        prob_gt (torch.Tensor): (b, 3, 5) GT
        torch.float32
    '''
    # reg loss
    alpha = 0.3
    huber = torch.nn.SmoothL1Loss()
    reg_loss = huber(scale_p, scale_gt)
    ce = torch.nn.CrossEntropyLoss() 
    prob_loss = ce(prob_p, prob_gt)

    se_loss = alpha*reg_loss + (1-alpha)*prob_loss

    return se_loss

def distillation_loss(z_1, z_2):
    r'''
    Divergence loss for latent feature 

    Args:
        z_1 (torch.Tensor): (b, 8192, l_dim) Student Model
        z_2 (torch.Tensor): (b, 8192, l_dim) Teacher Model
    '''
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    input = F.log_softmax(z_1, dim=1)
    target = F.softmax(z_2, dim=1)

    output = kl_loss(input, target)

    return output

def sim_loss(pcs1, pcs2):
    r'''
    Similarity loss

    Args:
        pcs1 (torch.Tensor): (b, N, 3)
        pcs2 (torch.Tensor): (b, N, 3)
    '''
    l2_loss = torch.nn.MSELoss()
    a_mean = torch.mean(pcs1, dim=1).unsqueeze(1) # b 1 3
    b_mean = torch.mean(pcs2, dim=1).unsqueeze(1) # b 1 3
    a_var = torch.var(pcs1, dim=1).unsqueeze(1) # b 1 3
    b_var = torch.var(pcs2, dim=1).unsqueeze(1) # b 1 3

    mean = l2_loss(a_mean, b_mean)
    var = l2_loss(a_var, b_var)

    sim = mean + var

    return torch.mean(sim)

def box_iou(pred_box, gt_box):
    r'''
    :pred_box: (B,60,7) -> Tensor
    :gt_box: (B,N,6) -> Tensor
    :return: positive_boxes
    '''
    pred_box_loc = torch.cat((pred_box[...,:3]-0.5*pred_box[...,3:6], pred_box[...,:3]+0.5*pred_box[...,3:6]), dim=-1)        # (B,60,6)
    gt_box_loc = torch.cat((gt_box[...,:3]-0.5*gt_box[...,3:6], gt_box[...,:3]+0.5*gt_box[...,3:6]), dim=-1)                  # (B,N,6)

    def box_volume(box):
        '''
        :box: (B,X,6)
        '''
        return (box[...,3]-box[...,0])*(box[...,4]-box[...,1])*(box[...,5]-box[...,2])                                # (B,X)
    
    pred_volume = box_volume(pred_box_loc)
    gt_volume = box_volume(gt_box_loc)

    lt = torch.max(gt_box_loc[:, :, None, :3], pred_box_loc[:, None, :, :3])                                          # (B,N,60,3)
    rb = torch.min(gt_box_loc[:, :, None, 3:], pred_box_loc[:, None, :, 3:])                                          # (B,N,60,3)

    inter = (rb - lt).clamp(min=0).prod(3)                                                                            # (B,N,60)
    iou = inter / (gt_volume[:, :, None] + pred_volume[:,None,:] - inter)                                             # (B,N,60)
    
    temp = iou
    pred_box_temp = pred_box
    max_box = []
    for i in range(iou.shape[1]):
        gt_temp = temp[0,i,:]
        posi_temp = torch.argmax(gt_temp,dim=-1)
        max_box.append(pred_box_temp[:,posi_temp])

        temp = torch.cat((temp[...,:posi_temp], temp[...,posi_temp+1:]), dim=-1)
        pred_box_temp = torch.cat((pred_box_temp[:,:posi_temp], pred_box_temp[:,posi_temp+1:]), dim=1)
    
    posi = torch.stack(max_box).permute(1,0,2)
    nega = pred_box_temp

    return posi, nega, iou

def bbox_loss(pred_box, gt_box):
    r'''
    :pred_box: (B,60,7)
    :gt_box: (B,N,6)
    :gt_num: (B), Num of positive
    :mask: (B,N,6)
    :return: prediction loss
    '''
    beta_loc = 2.0
    beta_cls = 1.0
    alpha = 0.25
    reg_loss = nn.SmoothL1Loss(reduce=None, reduction='none')

    N_pos = gt_box.shape[1]                                                                        # (B,1,1)
    positive, negative, iou = box_iou(pred_box, gt_box)                                                                            # (B,N,7)  
    print(iou.size())
    N_neg = negative.shape[1]

    box_xy = positive[...,:2]
    box_diag = torch.norm(box_xy,p=2,dim=-1)                                                                          # (B,N)
    dx = (gt_box[...,0]-positive[...,0])/box_diag
    dy = (gt_box[...,1]-positive[...,1])/box_diag
    dz = (gt_box[...,2]-positive[...,2])/gt_box[...,5]
    dw = torch.log((gt_box[...,3]+1e-4)/(positive[...,3]+1e-4))
    dl = torch.log((gt_box[...,4]+1e-4)/(positive[...,4]+1e-4))
    dh = torch.log((gt_box[...,5]+1e-4)/(positive[...,5]+1e-4))

    db = (torch.stack([dx,dy,dz,dw,dl,dh]).permute(1,2,0))                                                     # (B,N,6)
    loc_loss = torch.sum(reg_loss(db,torch.zeros_like(db)))/N_pos

    conf = positive[...,-1]
    cls_loss = -alpha*torch.sum(torch.pow((torch.ones_like(conf)-conf), 2)*torch.log(conf+1e-4))/N_pos

    nega_conf = negative[...,-1]
    neg_loss = -(1.0-alpha)*torch.sum(torch.pow((nega_conf), 2)*torch.log(1.0-nega_conf+1e-4))/N_neg
    
    loss = beta_loc*loc_loss + beta_cls*(cls_loss + neg_loss)

    return loss