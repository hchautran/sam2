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

    min_val= np.min(activations, axis=0)
    max_val = np.max(activations, axis=0)
    mean= np.max(activations, axis=0)
    p1 = np.percentile(activations, 1, axis=0)
    p99 = np.percentile(activations, 99, axis=0)
    p25 = np.percentile(activations, 25, axis=0)
    p75 = np.percentile(activations, 75, axis=0)

    return Distribution(
        n_channels=activations.shape[-1],
        dist_name=title,
        mean=mean,
        min=min_val,
        max=max_val,
        p1=p1,
        p99=p99,
        p25=p25,
        p75=p75
    )


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
        self.bins = 200
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
                        ObserverBase.dictionary['activation'][self.name_dict[module_type][sub_model][layer_idx]]=torch.zeros((self.bins,)).cpu().detach().numpy()

    def register_tensor_distribution_hook(self, model:nn.Module, min=-0.05, max=0.05):

        def pre_hook(module, input, name):
            input_tensor = input[0]
            bins = torch.linspace(min, max, self.bins)
            hist = torch.histc(input_tensor.float(), bins=bins.shape[0], min=min, max=max)
            hist = (hist/hist.sum()).cpu().detach().numpy()
            ObserverBase.dictionary['activation'][name] += hist

        def post_hook(module, input, output, name):
            output_tensor = output
            bins = torch.linspace(min, max, self.bins)
            hist = torch.histc(output_tensor.float(), bins=bins.shape[0], min=min, max=max)
            hist = (hist/hist.sum()).cpu().detach().numpy()
            ObserverBase.dictionary['activation'][name] += hist


        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in ObserverBase.dictionary['activation'].keys():
                # module.register_forward_pre_hook(partial(pre_hook,name=name))
                module.register_forward_hook(partial(post_hook,name=name))

    def register_token_distribution_hook(self, model:nn.Module):
        def pre_hook(module, input, name):
            pass

        def post_hook(module, input, output, name):
            pass

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in ObserverBase.dictionary['activation'].keys():
                # module.register_forward_pre_hook(partial(pre_hook,name=name))
                module.register_forward_hook(partial(post_hook,name=name))
    
    def register_channel_distribution_hook(self, model:nn.Module):

        def pre_hook(module, input, name):
            input_tensor = input[0].cpu().detach().numpy()
            ObserverBase.dictionary['activation'][name]=get_activation_distribution( input_tensor, title=name )


        def post_hook(module, input, output, name):
            output_tensor = output.cpu().detach().numpy()
            ObserverBase.dictionary['activation'][name]=get_activation_distribution( output_tensor, title=name )    


        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in ObserverBase.dictionary['activation'].keys():
                # module.register_forward_pre_hook(partial(pre_hook,name=name))
                module.register_forward_hook(partial(post_hook,name=name))

    def get_activation_distribution(self, name):
        return get_activation_distribution(activations=ObserverBase.dictionary['activation'][name], title=name)



# %%
#build model
checkpoint = './checkpoints/sam2.1_hiera_small.pt'
model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'
image_dir = './notebooks/images/truck.jpg'
video_dir = './notebooks/videos/bedroom'
def get_tensor_distribution(use_vid:bool, min=-2.0, max=2.0 ):
    observer = ActivationObserver(module_list=(nn.Linear,))
    min_val = -2.0
    max_val = 2.0
    if use_vid:
        predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        observer.init_activation_cache(predictor.image_encoder.trunk, layer_idxes=[0,12])
        observer.init_activation_cache(predictor.sam_mask_decoder, layer_idxes=[0])
        observer.register_tensor_distribution_hook(predictor, min=min_val, max=max_val)
        observer.inference_video(predictor=predictor, show_video=False)
    else:
        sam2 = build_sam2(model_cfg, checkpoint)
        predictor = SAM2ImagePredictor(sam2)
        observer.init_activation_cache(sam2.image_encoder.trunk, layer_idxes=[0,12])
        observer.init_activation_cache(sam2.memory_attention, layer_idxes=[0,1])
        observer.init_activation_cache(sam2.sam_mask_decoder, layer_idxes=[0])
        observer.register_tensor_distribution_hook(predictor.model, min=min_val, max=max_val)
        observer.inference_image(predictor=predictor, show_image=False)

    names = list(ObserverBase.dictionary['activation'].keys())
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(names):
        bins = torch.linspace(min, max, observer.bins).numpy()
        hist = ObserverBase.dictionary['activation'][name].squeeze()
        plt.bar(bins, hist, width=(bins[1] - bins[0]).item(), alpha=0.6, label=name)
        plt.title(name)
        plt.show()
    observer.clear_hook()
    observer.clear_dict()


def get_channel_distribution(use_vid:bool ):
    observer = ActivationObserver(module_list=(nn.Linear,))
    min_val = -2.0
    max_val = 2.0
    if use_vid:
        predictor = build_sam2_video_predictor(model_cfg, checkpoint)
        observer.init_activation_cache(predictor.image_encoder.trunk, layer_idxes=[0,12])
        observer.init_activation_cache(predictor.sam_mask_decoder, layer_idxes=[0])
        observer.register_channel_distribution_hook(predictor)
        observer.inference_video(predictor=predictor, show_video=False)
    else:
        sam2 = build_sam2(model_cfg, checkpoint)
        predictor = SAM2ImagePredictor(sam2)
        observer.init_activation_cache(sam2.image_encoder.trunk, layer_idxes=[0,12])
        observer.init_activation_cache(sam2.memory_attention, layer_idxes=[0,1])
        observer.init_activation_cache(sam2.sam_mask_decoder, layer_idxes=[0])
        observer.register_channel_distribution_hook(predictor.model)
        observer.inference_image(predictor=predictor, show_image=False)

    names = list(ObserverBase.dictionary['activation'].keys())
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(names):
        print(f"Distribution for {name}")
        _, ax = plt.subplots(1,1, figsize=(10, 6))
        ObserverBase.dictionary['activation'][name].plot_channel_distribution(ax)
    observer.clear_hook()
    observer.clear_dict()


# %%
# get_tensor_distribution(use_vid=True)
get_channel_distribution(use_vid=False)


# %%

# %%

