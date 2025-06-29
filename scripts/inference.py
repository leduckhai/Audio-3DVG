
import os
import sys
import json
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))
from lib.config import CONF
from lib.dataset import ScannetReferenceDataset
from data.scannet.model_util_scannet import ScannetDatasetConfig
from models.audio_3dvg import Audio3DVG
from utils.box_util import get_3d_box, box3d_iou
from utils.util import construct_bbox_corners

SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val_with_id.json")))
MAX_NUM_OBJECT = 16

def get_dataloader(args, scanrefer, all_scene_list, split):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer,
        scanrefer_all_scene=all_scene_list,
        split=split,
        args=CONF
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    return dataset, dataloader


def get_model(args):
    # load model
    model = Audio3DVG(
        args=args
    )

    print(args.use_checkpoint)
    path = os.path.join(args.use_checkpoint, "best_model.pth")
    print('checkpoint: ', args)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.cuda()

    return model


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer():
    scene_list = sorted(list(set([data["scene_id"] for data in SCANREFER_VAL])))
    scanrefer = [data for data in SCANREFER_VAL if data["scene_id"] in scene_list]

    return scanrefer, scene_list

def get_predict(data_dict, config):

    ref_center_label = data_dict["ref_center_label"].detach().cpu().numpy()
    ref_heading_class_label = data_dict["ref_heading_class_label"].detach().cpu().numpy()
    ref_heading_residual_label = data_dict["ref_heading_residual_label"].detach().cpu().numpy()
    ref_size_class_label = data_dict["ref_size_class_label"].detach().cpu().numpy()
    ref_size_residual_label = data_dict["ref_size_residual_label"].detach().cpu().numpy()

    ref_gt_obb = config.param2obb_batch(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                        ref_size_class_label, ref_size_residual_label)
    
    bts_candidate_mask = data_dict['bts_candidate_mask']
    bts_candidate_obbs = data_dict["bts_candidate_obbs"] # B x 16 x 6
    batch_size = bts_candidate_obbs.shape[0]
    scores = data_dict["score"]
    scores = scores.reshape(batch_size, MAX_NUM_OBJECT)

    pred_bboxes = []
    gt_bboxes = []
    audio_ids = []
    scene_ids = []

    for i in range(batch_size):  
        pred_obb = bts_candidate_obbs[i].cpu().numpy()
        num_filtered_obj = torch.sum(bts_candidate_mask[i])
        if num_filtered_obj == 0:
            continue
        elif num_filtered_obj == 1:
            pred_obb = pred_obb[0]
        else:
            score = scores[i].reshape(-1)[:num_filtered_obj]
            cluster_pred = torch.argmax(score[:num_filtered_obj], dim=0)
            pred_obb = bts_candidate_obbs[i][cluster_pred].cpu().numpy()
        audio_id = data_dict['id'][i]
        scene_id = data_dict['scene_id'][i]
        gt_obb = ref_gt_obb[i]
        # pred_bbox = get_3d_box(pred_obb[3:6], 0, pred_obb[0:3]) # 8 x 3
        # gt_bbox = get_3d_box(gt_obb[3:6], gt_obb[6], gt_obb[0:3]) # 8 x 3
        # pred_bbox = construct_bbox_corners(pred_obb[0:3], pred_obb[3:6])
        # gt_bbox = construct_bbox_corners(gt_obb[0:3], gt_obb[3:6])

        scene_ids.append(scene_id)
        audio_ids.append(audio_id)
        pred_bboxes.append(pred_obb)
        gt_bboxes.append(gt_obb)
    return audio_ids, scene_ids, pred_bboxes, gt_bboxes


def inference(args):

    os.makedirs(os.path.join(args.use_checkpoint, 'results'), exist_ok=True)
    DC = ScannetDatasetConfig()
    scanrefer, scene_list = get_scanrefer()

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val")

    # model
    model = get_model(args)

    # random seeds
    seeds = [args.manual_seed]

    for seed in seeds:
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)

        for data in tqdm(dataloader):
            for key in data:
                if key in ['object_cat','point_min', 'point_max', 'mlm_label',
                            'ref_center_label', 'ref_size_residual_label']:
                    data[key] = data[key].cuda()

            # feed
            data = model(data)
            scan_ids, scene_ids, pred_bboxes, gt_bboxes = get_predict(data, DC)
            if len(scan_ids) > 0:
                for scan_id, scene_id, pred_bbox, gt_bbox in zip(scan_ids, scene_ids, pred_bboxes, gt_bboxes):
                    result_path = os.path.join(args.use_checkpoint, 'results', "{}.json".format(scan_id))
                    r = {'scene_id': scene_id, 'pred': [j.tolist() for j in pred_bbox], 'gt': [j.tolist() for j in gt_bbox]}
                    with open(result_path, "w") as f:
                        json.dump(r, f, indent=4) 

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = CONF.gpu
    inference(CONF)