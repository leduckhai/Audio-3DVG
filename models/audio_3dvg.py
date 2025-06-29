import sys
import os
import importlib
import models
import torch.nn as nn

importlib.reload(models)

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
sys.path.append(os.path.join(os.getcwd(), "models"))  # HACK add the lib folder


class Audio3DVG(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

        # --------- AUDIO ENCODING ---------
        module = importlib.import_module(args.audio_module)
        self.audio = module.AudioModule()


        # --------- FEATURE ENCODING ---------
        module = importlib.import_module(args.feature_module)
        self.feature = module.FeatureModule()
        # --------- FEATURE ENCODING ---------
        module = importlib.import_module(args.attention_module)
        self.attention = module.AttentionModule()


    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds,
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        ### audio module
        data_dict = self.audio(data_dict) # B x 1 x 1024
        data_dict = self.feature(data_dict)
        ### attention module
        data_dict = self.attention(data_dict)

        return data_dict
