import os
import sys
import torch
import numpy as np
import open3d as o3d

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from utils.box_util import get_3d_box, box3d_iou
from utils.util import construct_bbox_corners

MAX_NUM_OBJECT = 16



def get_eval(data_dict, config):
    """ Loss functions
    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
        post_processing: config dict
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    # """
    m_cluster_labels = data_dict['cluster_label']

    ref_center_label = data_dict["ref_center_label"].detach().cpu().numpy()
    ref_heading_class_label = data_dict["ref_heading_class_label"].detach().cpu().numpy()
    ref_heading_residual_label = data_dict["ref_heading_residual_label"].detach().cpu().numpy()
    ref_size_class_label = data_dict["ref_size_class_label"].detach().cpu().numpy()
    ref_size_residual_label = data_dict["ref_size_residual_label"].detach().cpu().numpy()

    ref_gt_obb = config.param2obb_batch(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                        ref_size_class_label, ref_size_residual_label)

    # MY EVAL IMPLEMENTATION
    
    bts_candidate_mask = data_dict['bts_candidate_mask']
    bts_candidate_obbs = data_dict["bts_candidate_obbs"]
    batch_size = bts_candidate_obbs.shape[0]
    scores = data_dict["score"] # B x 8 x 1
    scores = scores.reshape(batch_size, MAX_NUM_OBJECT)
    # batch_size = bts_candidate_obbs.shape[0]
    m_ref_acc = []
    m_ious = []
    m_pred_bboxes = []
    m_gt_bboxes = []
    m_multiple = []
    m_others = []


    m_num_missed = 0
    for ii in range(batch_size):  
        m_pred_obb = bts_candidate_obbs[ii].cpu().numpy()
        m_num_filtered_obj = torch.sum(bts_candidate_mask[ii])
        # print("m_num_filtered_obj: ", m_num_filtered_obj)
        if m_num_filtered_obj == 0:
            m_pred_obb = np.zeros(7)
            m_num_missed += 1
        elif m_num_filtered_obj == 1:
            m_pred_obb = m_pred_obb[0]
        else:
            m_score = scores[ii].reshape(-1)[:m_num_filtered_obj]
  
            m_cluster_pred = torch.argmax(m_score, dim=0)

            m_target = m_cluster_labels[ii]

            if m_target == m_cluster_pred:
                m_ref_acc.append(1.)
            else:
                m_ref_acc.append(0.)
            m_pred_obb = bts_candidate_obbs[ii][m_cluster_pred].cpu().numpy()

        m_gt_obb = ref_gt_obb[ii]
        m_pred_bbox = get_3d_box(m_pred_obb[3:6], 0, m_pred_obb[0:3])
        m_gt_bbox = get_3d_box(m_gt_obb[3:6], m_gt_obb[6], m_gt_obb[0:3])
        m_iou = box3d_iou(m_pred_bbox, m_gt_bbox)
        m_ious.append(m_iou)

        # NOTE: get_3d_box() will return problematic bboxes
        m_pred_bbox = construct_bbox_corners(m_pred_obb[0:3], m_pred_obb[3:6])
        m_gt_bbox = construct_bbox_corners(m_gt_obb[0:3], m_gt_obb[3:6])

        if m_num_filtered_obj <= 1:
            # m_ref_acc.append(1.)
            if m_iou > 0.25:
                m_ref_acc.append(1.)
            else:
                m_ref_acc.append(0.)

        m_pred_bboxes.append(m_pred_bbox)
        m_gt_bboxes.append(m_gt_bbox)

        # construct the multiple mask
        m_multiple.append(data_dict["unique_multiple"][ii].item())

        # construct the others mask
        flag = 1 if data_dict["object_cat"][ii] == 17 else 0
        m_others.append(flag)


    # data_dict["seg_acc"] = torch.ones(1)[0].cuda()
    data_dict['ref_acc'] = m_ref_acc
    data_dict["ref_iou"] = m_ious
    data_dict["ref_iou_rate_0.25"] = np.array(m_ious)[np.array(m_ious) >= 0.25].shape[0] / np.array(m_ious).shape[0]
    data_dict["ref_iou_rate_0.5"] = np.array(m_ious)[np.array(m_ious) >= 0.5].shape[0] / np.array(m_ious).shape[0]

    # data_dict["seg_acc"] = torch.ones(1)[0].cuda()
    data_dict["ref_multiple_mask"] = m_multiple
    data_dict["ref_others_mask"] = m_others
    data_dict["pred_bboxes"] = m_pred_bboxes
    data_dict["gt_bboxes"] = m_gt_bboxes
    data_dict["num_miss"] = m_num_missed

    return data_dict