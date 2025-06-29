import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from utils.box_util import get_3d_box_batch, box3d_iou_batch

MAX_NUM_OBJECT = 16
loss_type = 'CE'


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, gamma=5, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction
        self.soft_plus = nn.Softplus()

    def forward(self, score, label):
        score *= self.gamma
        sim = (score*label).sum()
        neg_sim = score*label.logical_not()
        neg_sim = torch.logsumexp(neg_sim, dim=0) # soft max
        loss = torch.clamp(neg_sim - sim + self.margin, min=0).sum()
        return loss

class TargetClassificationLoss(nn.Module):
    def __init__(self):
        super(TargetClassificationLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, pred, gt):
        return self.criterion(pred, gt)


def get_loss(data_dict, config):
    """ Loss functions
    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    if loss_type == 'CE':
        criterion = TargetClassificationLoss().cuda()
    else:
        criterion = ContrastiveLoss(margin=0.2, gamma=5)

    ref_center_label = data_dict["ref_center_label"].detach().cpu().numpy()
    ref_heading_class_label = data_dict["ref_heading_class_label"].detach().cpu().numpy()
    ref_heading_residual_label = data_dict["ref_heading_residual_label"].detach().cpu().numpy()
    ref_size_class_label = data_dict["ref_size_class_label"].detach().cpu().numpy()
    ref_size_residual_label = data_dict["ref_size_residual_label"].detach().cpu().numpy()

    ref_gt_obb = config.param2obb_batch(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                        ref_size_class_label, ref_size_residual_label)
    ref_gt_bbox = get_3d_box_batch(ref_gt_obb[:, 3:6], ref_gt_obb[:, 6], ref_gt_obb[:, 0:3])


    cluster_label = []

    bts_candidate_mask = data_dict["bts_candidate_mask"]
    ref_loss = torch.zeros(1).cuda().requires_grad_(True)
    bts_candidate_obbs = data_dict["bts_candidate_obbs"]
    batch_size = bts_candidate_obbs.shape[0]
    scores = data_dict["score"].reshape(batch_size, MAX_NUM_OBJECT)
    
    if loss_type == 'CE':
        label = []
        for i in range(batch_size):  
            m_num_filtered_obj = torch.sum(bts_candidate_mask[i]).cpu()
            if m_num_filtered_obj == 0 or m_num_filtered_obj == 1:
                cluster_label.append(0)
                label.append(-100)
                continue
            m_label = np.zeros(MAX_NUM_OBJECT)
            candidate_obbs = bts_candidate_obbs[i].cpu().numpy() 
            pred_bbox = get_3d_box_batch(candidate_obbs[:, 3:6], np.zeros(MAX_NUM_OBJECT), candidate_obbs[:, 0:3])
            ious = box3d_iou_batch(pred_bbox, np.tile(ref_gt_bbox[i], (MAX_NUM_OBJECT, 1, 1)))
            label.append(ious.argmax())
            print('use: ', np.max(ious))
            m_label[ious.argmax()] = 1
            m_label = torch.FloatTensor(m_label).cuda()
            cluster_label.append(ious.argmax())

        label = np.array(label)
        label = torch.from_numpy(label).long().cuda()
        ref_loss = ref_loss + criterion(scores, label)

    if loss_type == 'CL':
        num_train_element = 0
        for i in range(batch_size):  
            m_num_filtered_obj = torch.sum(bts_candidate_mask[i])
            if m_num_filtered_obj == 0:
                cluster_label.append(0)
                continue
            if m_num_filtered_obj == 1:
                cluster_label.append(0)
                continue

            candidate_obbs = bts_candidate_obbs[i, :m_num_filtered_obj].cpu().numpy() 
            score = scores[i, :m_num_filtered_obj]   

            pred_bbox = get_3d_box_batch(candidate_obbs[:, 3:6], np.zeros(m_num_filtered_obj), candidate_obbs[:, 0:3])
            ious = box3d_iou_batch(pred_bbox, np.tile(ref_gt_bbox[i], (m_num_filtered_obj.cpu(), 1, 1)))

            label = ious.argmax()
            cluster_label.append(label)
            label = np.array(label).reshape(1,)
            label = torch.from_numpy(label).long().cuda()
            score = score.reshape(1, m_num_filtered_obj)
            ref_loss = ref_loss + criterion(score, label)
            num_train_element += 1


    cluster_label = torch.FloatTensor(cluster_label).cuda()
    ref_loss = ref_loss / batch_size
    data_dict['ref_loss'] = ref_loss
    data_dict['loss'] =  ref_loss
    data_dict['cluster_label'] = cluster_label
    return data_dict
