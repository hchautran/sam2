# %%
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from observer import ObserverBase


# %%

class EnergyObserver(ObserverBase):
    def __init__(self, module_list):
        super().__init__(module_list)

    def cal_energy(self, X:torch.Tensor, margin=0.5):
        """
        Calculate the energy of the input tensor.
        """
        X = F.normalize(X.reshape(-1, X.shape[-1]), dim=-1, p=2)
        score_map = F.elu(X @ X.transpose(-1, -2) - margin)
        scores =  score_map.mean(-1)
        return scores


    def register_energy_hook(self, model:nn.Module):
        self.dictionary['energy'] = defaultdict(list)

        def pre_hook(module, input, name):
            # calculate energy
            if input[0].shape[-2] <= 64:
                self.dictionary['energy'][name].append(self.cal_energy(input[0]).cpu().detach().numpy())

        self.register_hooks(model, pre_hook=pre_hook)

    def show_video_energy_map(self, layer_idx, frame_idx):
        names = list(ObserverBase.dictionary['energy'].keys())
        energies = list(ObserverBase.dictionary['energy'][names[layer_idx]])
        width = int(math.sqrt(energies[frame_idx].shape[0]))
        energies_map= energies[frame_idx].reshape(width, -1)
        plt.imshow(energies_map)
        plt.colorbar()
        plt.show()
# %%

#build model
from sam2.modeling.backbones.hieradet import MultiScaleBlock
checkpoint = './checkpoints/sam2.1_hiera_large.pt'
model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
# video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)
sam2 = build_sam2(model_cfg, checkpoint)
predictor = SAM2ImagePredictor(sam2)
# inference video
observer = EnergyObserver(module_list=(MultiScaleBlock,))
observer.register_energy_hook(predictor.model)
# observer.inference_video(predictor=predictor, show_video=True)
observer.inference_image(predictor, show_image=True, image_dir='./notebooks/images/truck.jpg')
observer.clear_hook()

# %%
observer.show_video_energy_map(layer_idx=35, frame_idx=0)
# observer.clear_dict()
