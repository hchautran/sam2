from dill.tests.test_classdef import n
# %% import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict

from torch.nn.modules import linear
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.memory_attention import MemoryAttention
from sam2.modeling.sam.mask_decoder import MaskDecoder
from sam2.modeling.memory_encoder import MemoryEncoder
from typing import Tuple, List
from functools import partial
from observer import ObserverBase
import seaborn as sns
import numpy as np
# %%
class Distribution():
    def __init__(self, n_channels:int, dist_name:str ,mean:np.ndarray, max:np.ndarray, min:np.ndarray, p75:np.ndarray, p25:np.ndarray, p99:np.ndarray, p1:np.ndarray):
        self.n_channels = n_channels
        self.mean = mean
        self.max = max
        self.min = min
        self.p75 = p75
        self.p25 = p25
        self.p99 = p99
        self.p1 = p1
        self.dist_name = dist_name


    def plot_channel_distribution(self, ax):
        x = np.arange(self.n_channels)
        ax.plot(x, self.mean, color='black', linewidth=-1)
        ax.plot(x, self.min, color='skyblue', linewidth=-1)
        ax.plot(x, self.max, color='skyblue', linewidth=-1, label='Min/Max')
        ax.plot(x, self.p1, color='red', linewidth=1)
        ax.plot(x, self.p99, color='red', linewidth=1, label='1/99 Percentile')
        ax.plot(x, self.p25, color='orange', linewidth=1)
        ax.plot(x, self.p75, color='orange', linewidth=1, label='25/75 Percentile')
        ax.set_title(self.dist_name)
        ax.set_xlabel('Hidden dimension index')
        ax.set_ylabel('Activation value')


def get_activation_distribution(activations:np.ndarray, title:str) -> Distribution:
# Compute percentiles along the sample axis (axis=-1)
    activations = np.reshape(activations, (-1, activations.shape[-1]))

    min= np.min(activations, axis=0)
    max = np.max(activations, axis=0)
    mean= np.max(activations, axis=0)
    p1 = np.percentile(activations, 1, axis=0)
    p99 = np.percentile(activations, 99, axis=0)
    p25 = np.percentile(activations, 25, axis=0)
    p75 = np.percentile(activations, 75, axis=0)

    return Distribution(
        n_channels=activations.shape[-1],
        dist_name=title,
        mean=mean,
        min=min,
        max=max,
        p1=p1,
        p99=p99,
        p25=p25,
        p75=p75
    )


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
            'MLP_up': 'mlp.layers.0',
            'MLP_down': 'mlp.layers.1'
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
            'MLP_up': 'mlp.layers.0',
            'MLP_down': 'mlp.layers.1'
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
            'MLP_up': 'mlp.layers.0',
            'MLP_down': 'mlp.layers.1',
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






class ActivationObserver(ObserverBase):

    def __init__(self, module_list:Tuple):
        super().__init__(module_list)
        self.IMAGE_ENCODER = 'Image Encoder'
        self.MEMORY_ATTENTION = 'Memory Attention'
        self.MASK_DECODER = 'Mask Decoder'
        self.MEMORY_ENCODER = 'Mask Encoder'
        self.name_dict= {
            self.IMAGE_ENCODER:  defaultdict(list),
            self.MEMORY_ATTENTION: defaultdict(list),
            self.MASK_DECODER: defaultdict(list),
            self.MEMORY_ENCODER: defaultdict(list)
        }

    def get_linear_name(self, module:nn.Module):
        sub_modules = get_submodule_names(module)
        model = get_model_name(module)
        if  isinstance(module, Hiera):
            num_layers=len(module.blocks)
            for layer_idx in range(num_layers):
                for key in sub_modules.keys():
                    self.name_dict[self.IMAGE_ENCODER][f'{key}'].append(f'{model}.{layer_idx}.{sub_modules[key]}')
        elif isinstance(module, MemoryAttention):
            num_layers=len(module.layers)
            for layer_idx in range(num_layers):
                for key in sub_modules.keys():
                    self.name_dict[self.MEMORY_ATTENTION][f'{key}'].append(f'{model}.{layer_idx}.{sub_modules[key]}')
        elif isinstance(module, MaskDecoder):
            num_layers=len(module.transformer.layers)
            for layer_idx in range(num_layers):
                for key in sub_modules.keys():
                    self.name_dict[self.MASK_DECODER][f'{key}'].append(f'{model}.{layer_idx}.{sub_modules[key]}')



    def init_activation_cache(self, module:nn.Module, layer_idxes:List[int]=[0]):
        # calculate energy
        if 'activation' not in ObserverBase.dictionary.keys():
            ObserverBase.dictionary['activation'] = {}
        self.get_linear_name(module)
        for module_type in self.name_dict.keys():
            if len(self.name_dict[module_type].keys()) > 0:
                for sub_model in self.name_dict[module_type]:
                    for layer_idx in layer_idxes:
                        ObserverBase.dictionary['activation'][self.name_dict[module_type][sub_model][layer_idx]]=None

    def register_distribution_hook(self, model:nn.Module, min=-0.05, max=0.05, layer_idxes:List[int]=[0]):
        # ObserverBase.dictionary['activation'] = {}

        def pre_hook(module, input, name):
            # dist = get_activation_distribution(activations=input[0].cpu().detach().numpy(), title=name)
            input_tensor = input[0]
            bins = torch.linspace(min, max, 200)

            hist = torch.histc(input_tensor.float(), bins=bins.shape[0], min=min, max=max)
            if ObserverBase.dictionary['activation'][name] is None:
                ObserverBase.dictionary['activation'][name] = hist.cpu().detach().numpy(),
            else:
                ObserverBase.dictionary['activation'][name] += hist.cpu().detach().numpy()

        def post_hook(module, input, output, name):
            # dist = get_activation_distribution(activations=input[0].cpu().detach().numpy(), title=name)
            output_tensor = output
            bins = torch.linspace(min, max, 200)
            hist = torch.histc(output_tensor.float(), bins=bins.shape[0], min=min, max=max)
            if ObserverBase.dictionary['activation'][name] is None:
                ObserverBase.dictionary['activation'][name] = hist.cpu().detach().numpy(),
            else:
                ObserverBase.dictionary['activation'][name] += hist.cpu().detach().numpy()


        for name, module in model.named_modules():

            if isinstance(module, nn.Linear) and name in self.dictionary['activation'].keys():
                # module.register_forward_pre_hook(partial(pre_hook,name=name))
                module.register_forward_hook(partial(post_hook,name=name))

    def get_activation_distribution(self, name):
        return get_activation_distribution(activations=ObserverBase.dictionary['activation'][name], title=name)



# %%
#build model
checkpoint = './checkpoints/sam2.1_hiera_large.pt'
model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
image_dir = './notebooks/images/truck.jpg'
video_dir = './notebooks/videos/bedroom'
video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)
sam2 = build_sam2(model_cfg, checkpoint)
image_predictor = SAM2ImagePredictor(sam2)
observer = ActivationObserver(module_list=(nn.Linear,))
observer.init_activation_cache(sam2.image_encoder.trunk)
# observer.init_activation_cache(sam2.memory_attention)
# observer.init_activation_cache(sam2.memory_encoder)
# observer.inference_video(predictor=predictor, show_video=True)
min =-1.0
max =1.0
observer.register_distribution_hook(video_predictor, min=min, max=max)
# observer.inference_image(predictor, show_image=False, image_dir=image_dir)
observer.inference_video(predictor=video_predictor, show_video=False)
# %%

names = list(observer.linear_names.keys())
plt.figure(figsize=(10, 6))
axes, figs = plt.subplots(1, 4, figsize=(10, 6*len(names)))
for i, name in enumerate(names):
    bins = torch.linspace(min, max, 200).numpy()
    hist = ObserverBase.dictionary['activation'][observer.linear_names[name]].squeeze()

    plt.bar(bins, hist, width=(bins[1] - bins[0]).item(), alpha=0.6, label=name)
plt.title('distribution')
plt.show()

# %%
observer.clear_hook()
observer.clear_dict()

# %%
