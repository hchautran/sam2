# %%
import os
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial
from typing import List


class ActivationRecoder:


    def __init__(self, module_list:List[int]):
        self.activations = defaultdict(list)
        self.module_list = module_list

    def register_hook(self, model: nn.Module):
        hooks = []

        def register_post_hook(module: nn.Module, input, output, module_name):
            print(name)
            self._save_activation(module_name, output)

        def register_pre_hook(module: nn.Module, input, module_name):
            self._save_activation(module_name, input[0])

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(
                    module.register_forward_hook(
                        partial(register_post_hook, name))
                )
                hooks.append(
                    module.register_forward_pre_hook(register_pre_hook, name))

        return hooks

    def _save_activation(self, module_name:str, activation:torch.Tensor):
        self.activations[module_name].append(activation.detach().cpu().numpy())

    def save(self, file_name: str, prefix=None):
        if prefix is None:
            prefix = os.getcwd()

        torch.save(self.activations, f'{prefix}/{file_name}.pth')
        print(f"Activations saved to {prefix}/{file_name}.pth")

# %%
import os
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt
checkpoint = './checkpoints/sam2.1_hiera_small.pt'
model_cfg = 'configs/sam2.1/sam2.1_hiera_s.yaml'
video_dir = './notebooks/videos/bedroom'

predictor = build_sam2_video_predictor(model_cfg, checkpoint)

frame_names = [
    p for p in os.listdir(video_dir) if os.path.splitext(p)[-1] in ['.jpg', '.jpeg', '.JPG', '.JPEG']
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
frame_idx = 0
plt.figure(figsize=(9,6))
plt.title(f'frame {frame_idx}')
plt.imshow(plt.imread(os.path.join(video_dir, frame_names[frame_idx])))


# %%
inference_state = predictor.init_state(video_path=video_dir)
# %%
print(type(predictor))
# %%
print(type(inference_state))
predictor.reset_state(inference_state)

# %%
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
import numpy as  np
from vis_utils import show_points, show_mask

def add_prompt(
    predictor:SAM2VideoPredictor, inference_state, points, ann_frame_idx:int=0, ann_obj_id:int=1, show_point:bool=False):

    labels = np.array([1, 1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )

    # show the results on the current (interacted) frame
    if show_point:
        plt.figure(figsize=(9, 6))
        plt.title(f"frame {ann_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
        show_points(points, labels, plt.gca())
        show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
# %%
module_list = (0, 10, 20)
profiler = ActivationRecoder(module_list)
profiler.profile(predictor)
add_prompt(
    predictor=predictor,
    inference_state=inference_state,
    points=np.array([[210, 350], [250, 220]], dtype=np.float32),
    ann_frame_idx=0,
    ann_obj_id=1,
    show_point=True
)
profiler.get_results()


# %%
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
# %%
print(profiler.get_results())
profiler.get_profile_plot()
profiler.clear()
