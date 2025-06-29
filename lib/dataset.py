import os
import sys
import torch
import numpy as np
from lib.config import CONF
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), "lib"))

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.pc_utils import random_sampling, rotx, roty, rotz
from data.scannet.model_util_scannet import ScannetDatasetConfig, rotate_aligned_boxes_along_axis
from torchsparse import SparseTensor
from torchsparse.utils import sparse_collate_fn, sparse_quantize

# data setting
DC = ScannetDatasetConfig()
# print(DC.nyu40id2class)
MAX_NUM_OBJ = 128
MAX_AUDIO_FRAME = 3000
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# data path
SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_META, "scannetv2-labels.combined.tsv")
MULTIVIEW_DATA = CONF.MULTIVIEW


class ScannetReferenceDataset(Dataset):

    def __init__(self, scanrefer, scanrefer_all_scene, split="train", args=None):

        self.args = args
        self.scanrefer = scanrefer
        self.scanrefer_all_scene = scanrefer_all_scene  # all scene_ids in scanrefer
        self.split = split
        self.num_points = args.num_points
        self.use_color = args.use_color
        self.use_height = args.use_height
        self.use_normal = args.use_normal
        self.use_multiview = args.use_multiview
        self.augment = args.use_augment if split == "train" else False

        if split == 'train':
            self.audio_path = 'data/contextual_train'
            self.nel_path = 'data/data_train.csv'
        else:
            self.audio_path = 'data/contextual_val'
            self.nel_path = 'data/data_val.csv'

        self.audio_class, self.nel_label = self._load_nel_label()

        self._load_data()

        # self.voxel_size_ap = np.array([self.args.voxel_size_ap, self.args.voxel_size_ap, self.args.voxel_size_ap])
        # self.voxel_size_glp = np.array([self.args.voxel_size_glp, self.args.voxel_size_glp, self.args.voxel_size_glp])

    def __len__(self):
        return len(self.scanrefer)

    def __getitem__(self, idx):
        scene_id = self.scanrefer[idx]["scene_id"]
        object_id = int(self.scanrefer[idx]["object_id"])
        object_name = " ".join(self.scanrefer[idx]["object_name"].split("_"))
        ann_id = int(self.scanrefer[idx]["ann_id"])
        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17

        # load voice feature
        audio_id = self.scanrefer[idx]['id']
        audio_feature = torch.load(os.path.join(self.audio_path, "{}.pt".format(audio_id))).unsqueeze(0)
        audio_length = audio_feature.shape[1]
        if audio_length < MAX_AUDIO_FRAME:
            padd_feature = torch.zeros((1, 3000 - audio_length, 768))
            audio_feature = torch.cat((audio_feature, padd_feature), dim=1)

        # load classification and NEL label
        audio_class =  self.audio_class[int(audio_id)]
        nel_label =  self.nel_label[int(audio_id)]

        # get pc
        mesh_vertices = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_aligned_vert.npy")  # axis-aligned
        instance_labels = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_ins_label_pg.npy")
        semantic_labels = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_sem_label_pg.npy")
        instance_bboxes = np.load(os.path.join(CONF.PATH.SCANNET_DATA, scene_id) + "_aligned_bbox.npy")

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:6] = (point_cloud[:, 3:6]) / 256.0
            pcl_color = point_cloud[:, 3:6]

        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))

        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))

        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))

        ref_box_label = np.zeros(MAX_NUM_OBJ)
        ref_center_label = np.zeros(3)

        ref_heading_class_label = 0
        ref_heading_residual_label = 0
        ref_size_class_label = 0
        ref_size_residual_label = np.zeros(3)

        if self.split != "test":
            num_bbox = instance_bboxes.shape[0] if instance_bboxes.shape[0] < MAX_NUM_OBJ else MAX_NUM_OBJ
            target_bboxes_mask[0:num_bbox] = 1
            target_bboxes[0:num_bbox, :] = instance_bboxes[:MAX_NUM_OBJ, 0:6]

            # ------------------------------- DATA AUGMENTATION ------------------------------
            if self.augment:
                if torch.rand(1).item() > 0.5:
                    # Flipping along the YZ plane
                    point_cloud[:, 0] = -1 * point_cloud[:, 0]
                    target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

                if torch.rand(1).item() > 0.5:
                    # Flipping along the XZ plane
                    point_cloud[:, 1] = -1 * point_cloud[:, 1]
                    target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

                # Rotation along X-axis
                rot_angle = (torch.rand(1).item() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotx(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "x")

                # Rotation along Y-axis
                rot_angle = (torch.rand(1).item() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = roty(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "y")

                # Rotation along up-axis/Z-axis
                rot_angle = (torch.rand(1).item() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
                rot_mat = rotz(rot_angle)
                point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
                target_bboxes = rotate_aligned_boxes_along_axis(target_bboxes, rot_mat, "z")

                # Translation
                point_cloud, target_bboxes = self._translate(point_cloud, target_bboxes)

            # NOTE: set size class as semantic class. Consider use size2class.
            class_ind = [DC.nyu40id2class[int(x)] for x in instance_bboxes[:num_bbox, -2]]
            size_classes[0:num_bbox] = class_ind
            size_residuals[0:num_bbox, :] = target_bboxes[0:num_bbox, 3:6] - DC.mean_size_arr[class_ind, :]

            # construct the reference target label for each bbox
            ref_box_label = np.zeros(MAX_NUM_OBJ)
            
            for i, gt_id in enumerate(instance_bboxes[:num_bbox, -1]): # -1 mean id of object in scene, equal to objecid
                if gt_id == object_id:
                    ref_box_label[i] = 1
                    ref_center_label = target_bboxes[i, 0:3]
                    ref_heading_class_label = angle_classes[i]
                    ref_heading_residual_label = angle_residuals[i]
                    ref_size_class_label = size_classes[i]
                    ref_size_residual_label = size_residuals[i]
        else:
            num_bbox = 1

        # ------------------------------- DATA HANDLING ------------------------------
        instance_points = []
        instance_class = []
        ins_obbs = []
        for i_instance in np.unique(instance_labels):
            ind = np.nonzero(instance_labels == i_instance)[0]
            ins_class = semantic_labels[ind[0]]
            if ins_class in DC.nyu40ids:
                x = point_cloud[ind]
                ins_class = DC.nyu40id2class[int(ins_class)]
                instance_class.append(ins_class)

                pc = x[:, :3]
                center = 0.5 * (pc.min(0) + pc.max(0))
                size = pc.max(0) - pc.min(0)
                ins_obb = np.concatenate((center, size, np.array([0])))
                ins_obbs.append(ins_obb)
                x = random_sampling(x, 1024)
                instance_points.append(x)


        data_dict = {}
        data_dict['id'] = audio_id
        data_dict['scene_id'] = scene_id
        data_dict['object_name'] = object_name
        data_dict['audio_feature'] = audio_feature
        data_dict['audio_length'] = audio_length
        data_dict['audio_class'] = audio_class
        data_dict['nel_label'] = nel_label

        # data_dict['lidar'] = pt
        data_dict['point_min'] = point_cloud.min(0)[:3]
        data_dict['point_max'] = point_cloud.max(0)[:3]
        data_dict['instance_labels'] = instance_labels.astype(np.int64)

        data_dict['instance_points'] = instance_points
        data_dict['instance_class'] = instance_class
        data_dict['instance_obbs'] = ins_obbs
        data_dict["point_clouds"] = point_cloud.astype(np.float32)  # scene point cloud

        
        data_dict["center_label"] = target_bboxes.astype(np.float32)[:, 0:3]  # (MAX_NUM_OBJ, 3) for GT box center XYZ
        data_dict["heading_class_label"] = angle_classes.astype(
            np.int64)  # (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
        data_dict["heading_residual_label"] = angle_residuals.astype(np.float32)  # (MAX_NUM_OBJ,)
        data_dict["size_class_label"] = size_classes.astype(
            np.int64)  # (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
        data_dict["size_residual_label"] = size_residuals.astype(np.float32)  # (MAX_NUM_OBJ, 3)
        data_dict["num_bbox"] = np.array(num_bbox).astype(np.int64)
        data_dict["scan_idx"] = np.array(idx).astype(np.int64)
        data_dict["pcl_color"] = pcl_color

        data_dict["ref_box_label"] = ref_box_label.astype(np.int64)  # 0/1 reference labels for each object bbox
        data_dict["ref_center_label"] = ref_center_label.astype(np.float32)
        data_dict["ref_heading_class_label"] = np.array(int(ref_heading_class_label)).astype(np.int64)
        data_dict["ref_heading_residual_label"] = np.array(int(ref_heading_residual_label)).astype(np.int64)
        data_dict["ref_size_class_label"] = np.array(int(ref_size_class_label)).astype(np.int64)
        data_dict["ref_size_residual_label"] = ref_size_residual_label.astype(np.float32)
        data_dict["object_id"] = np.array(int(object_id)).astype(np.int64)
        data_dict["ann_id"] = np.array(ann_id).astype(np.int64)
        data_dict["object_cat"] = np.array(object_cat).astype(np.int64)
        data_dict["unique_multiple"] = np.array(
            self.unique_multiple_lookup[scene_id][str(object_id)][str(ann_id)]).astype(np.int64)

        return data_dict

    def _get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys() # 0->17
        scannet2label = {label: i for i, label in enumerate(scannet_labels)} # 0->17

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label['others']
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]
        # print(raw2label)
        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup


    def _load_data(self):
        print("Loading data...")

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _translate(self, point_set, bbox):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        factor = (torch.rand(3) - 0.5).tolist()

        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor

        return point_set, bbox
    
    def _load_nel_label(self):
        df =  pd.read_csv(self.nel_path)
        return df['class'].to_numpy(), df['nel'].to_numpy()

    @staticmethod
    def collate_fn(inputs):
        outputs = sparse_collate_fn(inputs)
        return outputs
