# %% import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.memory_attention import MemoryAttention
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.memory_encoder import MemoryEncoder
from typing import Tuple, List

from observer import ObserverBase
import seaborn as sns
import numpy as np
# %%

def plot_percentile_bands(ax, activations:np.ndarray, title:str):
# Compute percentiles along the sample axis (axis=-1)
    min_vals = np.percentile(activations, -1, axis=0)
    max_vals = np.percentile(activations, 99, axis=0)
    p0 = np.percentile(activations, 1, axis=0)
    p98 = np.percentile(activations, 99, axis=0)
    p24 = np.percentile(activations, 25, axis=0)
    p74 = np.percentile(activations, 75, axis=0)
    x = np.arange(activations.shape[0])
    ax.plot(x, min_vals, color='skyblue', linewidth=0)
    ax.plot(x, max_vals, color='skyblue', linewidth=0, label='Min/Max')
    ax.plot(x, p0, color='red', linewidth=1)
    ax.plot(x, p98, color='red', linewidth=1, label='1/99 Percentile')
    ax.plot(x, p24, color='orange', linewidth=1)
    ax.plot(x, p74, color='orange', linewidth=1, label='25/75 Percentile')

    ax.set_title(title)
    ax.set_xlabel('Hidden dimension index')
    ax.set_ylabel('Activation value')

def plot_distribution(ax, values:np.ndarray, title:str):
    sns.histplot(values, bins=30, kde=True, stat='density', color='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()

def get_submodule_names(module:nn.Module)->dict:
    if isinstance(module, Hiera):
        return {
            'QKV': 'attn.qkv',
            'O': 'attn.proj',
            'MLP0': 'mlp.layers.0',
            'MLP1': 'mlp.layers.1'
        }
    elif isinstance(module, MemoryAttention):
        return {
            'self attn Q': 'self_attn.q_proj',
            'self attn k': 'self_attn.k_proj',
            'self attn V': 'self_attn.v_proj',
            'self attn O': 'self_attn.out_proj',
            'cross attn Q': 'cross_attn.q_proj',
            'cross attn K': 'cross_attn.k_proj',
            'cross attn V': 'cross_attn.v_proj',
            'cross attn O': 'cross_attn.out_proj',
            'MLP1': 'mlp.layers.0',
            'MLP2': 'mlp.layers.1'
        }
    elif isinstance(module, MaskDecoder):
        return {
            'self attn Q': 'self_attn.q_proj',
            'self attn K': 'self_attn.k_proj',
            'self attn V': 'self_attn.v_proj',
            'self attn O': 'self_attn.out_proj',
            'cross attn i2t Q': 'cross_attn_image_to_token.q_proj',
            'cross attn i2t K': 'cross_attn_image_to_token.k_proj',
            'cross attn i2t V': 'cross_attn_image_to_token.v_proj',
            'cross attn i2t O': 'cross_attn_image_to_token.out_proj',
            'cross attn t2i Q': 'cross_attn_token_to_image.q_proj',
            'cross attn t2i K': 'cross_attn_token_to_image.k_proj',
            'cross attn t2i V': 'cross_attn_token_to_image.v_proj',
            'cross attn t2i O': 'cross_attn_token_to_image.out_proj',
            'final attn t2i Q': 'final_attn_token_to_image.q_proj',
            'final attn t2i K': 'final_attn_token_to_image.k_proj',
            'final attn t2i V': 'final_attn_token_to_image.v_proj',
            'final attn t2i O': 'final_attn_token_to_image.out_proj',
            'MLP0': 'mlp.layers.0',
            'MLP1': 'mlp.layers.1',
        }
    else:
        return {}


def get_model_name(module:nn.Module):
    if isinstance(module, Hiera):
        return 'image_encoder.trunk.blocks'
    if isinstance(module, MemoryAttention):
        return 'memory_attention.layers'
    if isinstance(module, MaskDecoder):
        return 'sam_mask_decoder.transformer.layers'



def get_linear_name(module:nn.Module, layer_idxes:List[int])-> dict:
    linear_names = {}
    sub_modules = get_submodule_names(module)
    model = get_model_name(module)

    for layer_idx in layer_idxes:
        if  isinstance(module, Hiera):
            for key in sub_modules.keys():
                linear_names[f'Image Encoder {key} {layer_idx}'] = f'{model}.{layer_idx}.{sub_modules[key]}'
        elif isinstance(module, MemoryAttention):
            for key in sub_modules.keys():
                linear_names[f'Memory Attention {key} {layer_idx}'] = f'{model}.{layer_idx}.{sub_modules[key]}'
        elif isinstance(module, MaskDecoder):
            for key in sub_modules.keys():
                linear_names[f'Mask Decoder {key} {layer_idx}'] = f'{model}.{layer_idx}.{sub_modules[key]}'

    return linear_names


class ActivationObserver(ObserverBase):

    def __init__(self, module_list:Tuple):
        super().__init__(module_list)

    def init_activation_cache(self, module:nn.Module, layer_idxes:List):
        # calculate energy
        ObserverBase.dictionary['activation'] = {}
        print(ObserverBase.dictionary)

        self.linear_names = get_linear_name(module, layer_idxes)
        for name in self.linear_names:
            ObserverBase.dictionary['activation'][name]=defaultdict(list)

    def plot_tensor_level_distribution(self, layer_idx):
        for idx, name in enumerate(self.linear_names):
            activation =  ObserverBase.dictionary['activation'][name]
            _, axes = plt.subplots(1, len(self.linear_names), figsize=(5*len(self.linear_names), 5))
            plot_distribution(axes[idx], activation, title=f'{name}')



    def plot_channel_level_distribution(self, layer_idx):
        for idx, name in enumerate(self.linear_names):
            activation =  ObserverBase.dictionary['activation'][name]
            _, axes = plt.subplots(1, len(self.linear_names), figsize=(5*len(self.linear_names), 5))
            plot_percentile_bands(axes[idx], activation, title=f'{name}')

    def register_distribution_hook(self, model:nn.Module):
        self.dictionary['activation'] = defaultdict(list)

        def pre_hook(module, input, name):
            self.dictionary['activation'][name].append(input.cpu().detach().numpy())

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in self.dictionary['activation'].keys():
                module.register_forward_pre_hook(pre_hook)


# %%

#build model
checkpoint = './checkpoints/sam2.1_hiera_small.pt'
model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'
video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)
sam2 = build_sam2(model_cfg, checkpoint)
predictor = SAM2ImagePredictor(sam2)

observer = ActivationObserver(module_list=(nn.Linear,))
observer.init_activation_cache(sam2.image_encoder.trunk, [0, 10])

# observer.inference_video(predictor=predictor, show_video=True)
observer.register_distribution_hook(sam2)
observer.inference_image(predictor, show_image=False, image_dir='./notebooks/images/truck.jpg')
# observer.clear_hook()
# observer.clear_dict()
# %%
ObserverBase.dictionary['activation']
