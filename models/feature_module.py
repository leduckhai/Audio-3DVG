import torch
import torch.nn.functional as F
import torch.nn as nn
import json
import ast
import numpy as np
from models.obj_encoder import PcdObjEncoder
from omegaconf import OmegaConf
config = OmegaConf.load("config/audio-3dvg.yaml")

class FeatureModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.object_encoder = PcdObjEncoder(config.obj_encoder)
        with open("label2vect.json", "r") as f:
            self.text_encoder = json.load(f)
        self.MAX_NUM_OBJECT = 16
        self.TEXT_EMBEDDING_DIM = 50


    def forward(self, data_dict):
        batch_size = len(data_dict['instance_points'])
        bts_candidate_point = []
        bts_candidate_obbs = []
        bts_candidate_mask = []
        
        bts_relation_point = []
        bts_relation_obbs = []
        bts_relation_mask = []
        
        bts_candidate_label_embedding = []
        bts_relation_label_embedding = []
        bts_audio_feature = []
        
        for i in range(batch_size):
            candidate_point = []
            candidate_obbs = []
            candidate_mask = []
            candidate_label_embedding = []
        
            relation_point = []
            relation_obbs = []
            relation_mask = []
            relation_label_embedding = []
            instance_point = data_dict['instance_points'][i]
            instance_obb = data_dict['instance_obbs'][i]
            instance_class = data_dict['instance_class'][i]
            # num_obj = len(instance_point)

            audio_feature = data_dict['embedded_audio'][i]
            bts_audio_feature.append(audio_feature)
            
            audio_class = data_dict['audio_class'][i]
            nel_label = data_dict['nel_label'][i]
            nel_label = ast.literal_eval(nel_label)

            for idx, i_class in enumerate(instance_class):
                if i_class == audio_class:
                    candidate_point.append(instance_point[idx][:, :6].tolist())
                    candidate_obbs.append(instance_obb[idx][:6].tolist())
                    candidate_mask.append(1)
                    candidate_label_embedding.append(self.text_encoder[str(i_class)])
                elif i_class in nel_label:
                    relation_point.append(instance_point[idx][:, :6].tolist())
                    relation_obbs.append(instance_obb[idx][:6].tolist())
                    relation_mask.append(1)
                    relation_label_embedding.append(self.text_encoder[str(i_class)])
                else:
                    continue

            # VISUALIZATION
            # m_gt_obb = ref_gt_obb[i]
            # geometries = []
            # geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
            # import open3d as o3d
            # for iii in candidate_point:
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(np.array(iii)[:, :3])
            #     pcd.colors = o3d.utility.Vector3dVector(np.array(iii)[:, 3:6])
            #     geometries.append(pcd)
            # o3d.visualization.draw_geometries(geometries)

            if len(candidate_point) > self.MAX_NUM_OBJECT:
                candidate_point = candidate_point[:self.MAX_NUM_OBJECT]
                candidate_obbs = candidate_obbs[:self.MAX_NUM_OBJECT]
                candidate_mask = candidate_mask[:self.MAX_NUM_OBJECT]
                candidate_label_embedding = candidate_label_embedding[:self.MAX_NUM_OBJECT]

            if len(relation_point) > self.MAX_NUM_OBJECT:
                relation_point = relation_point[:self.MAX_NUM_OBJECT]
                relation_obbs = relation_obbs[:self.MAX_NUM_OBJECT]
                relation_mask = relation_mask[:self.MAX_NUM_OBJECT]
                relation_label_embedding = relation_label_embedding[:self.MAX_NUM_OBJECT]

            while len(candidate_point) < self.MAX_NUM_OBJECT:
                candidate_point.append(np.zeros((1024, 6)).tolist())
                candidate_obbs.append(np.zeros(6).tolist())
                candidate_mask.append(0)
                candidate_label_embedding.append(np.zeros(self.TEXT_EMBEDDING_DIM).tolist())
            while len(relation_point) < self.MAX_NUM_OBJECT:
                relation_point.append(np.zeros((1024, 6)).tolist())
                relation_obbs.append(np.zeros(6).tolist())
                relation_mask.append(0)
                relation_label_embedding.append(np.zeros(self.TEXT_EMBEDDING_DIM).tolist())


            bts_candidate_point.append(candidate_point)
            bts_candidate_obbs.append(candidate_obbs)
            bts_candidate_mask.append(candidate_mask)
            bts_candidate_label_embedding.append(candidate_label_embedding)

            bts_relation_point.append(relation_point)
            bts_relation_obbs.append(relation_obbs)
            bts_relation_mask.append(relation_mask)
            bts_relation_label_embedding.append(relation_label_embedding)

        
        bts_candidate_point = torch.tensor(bts_candidate_point).cuda()
        bts_relation_point = torch.tensor(bts_relation_point).cuda()

        bts_candidate_obbs = torch.tensor(bts_candidate_obbs).cuda() 
        bts_relation_obbs = torch.tensor(bts_relation_obbs).cuda()

        bts_candidate_mask = torch.tensor(bts_candidate_mask).cuda()
        bts_relation_mask = torch.tensor(bts_relation_mask).cuda() 

        bts_candidate_label_embedding = torch.tensor(bts_candidate_label_embedding).cuda()
        bts_relation_label_embedding = torch.tensor(bts_relation_label_embedding).cuda()

        bts_audio_feature = torch.stack(bts_audio_feature).cuda()
        object_encoding = self.object_encoder(torch.cat([bts_candidate_point, bts_relation_point], dim=1))

        target_representation = torch.cat((bts_candidate_obbs, bts_candidate_label_embedding, object_encoding[:, :self.MAX_NUM_OBJECT, :]), dim=2)
        relation_representation = torch.cat((bts_relation_obbs, bts_relation_label_embedding, object_encoding[:, self.MAX_NUM_OBJECT:, :]), dim=2)
        data_dict["target_representation"] = target_representation
        data_dict["relation_representation"] = relation_representation
        data_dict["bts_audio_feature"] = bts_audio_feature
        data_dict["bts_candidate_obbs"] = bts_candidate_obbs
        data_dict["bts_relation_obbs"] = bts_relation_obbs
        data_dict["bts_candidate_mask"] = bts_candidate_mask
        data_dict["bts_relation_mask"] = bts_relation_mask
        return data_dict