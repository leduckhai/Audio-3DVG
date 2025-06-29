import torch
import torch.nn as nn


class AudioModule(nn.Module):
    def __init__(self):
        super(AudioModule, self).__init__()

        self.gru = nn.GRU(
            input_size=768,
            hidden_size=512,
            batch_first=True,
            bidirectional=True
        )
        for name, param in self.gru.named_parameters():
            param.requires_grad = True
    
    def forward(self, data_dict):
        feats = data_dict["audio_feature"] # B x 1 x padd x 768
        lengths = data_dict['audio_length']
        batch_size = feats.shape[0]
        gru_out = []
        for i in range(batch_size):
            input_tensor = feats[i]
            input_tensor = input_tensor[0, :lengths[i], :].unsqueeze(0).cuda()
            _, hidden = self.gru(input_tensor)
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            gru_out.append(hidden)
        data_dict['embedded_audio'] = torch.stack(gru_out)
        return data_dict
        