# %%
import os
import torch
import torch.nn as nn
import torch.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial


class ActivationRecoder:
   def __init__(self):
      self.activations = defaultdict(list) 

   def register_hook(self, model: nn.Module):
      hooks = []

      def register_post_hook(module: nn.Module, input, output):
         module_name = module.__class__.__name__
         self._save_activation(module_name, output)


      def register_pre_hook(module: nn.Module, input):
         module_name = module.__class__.__name__
         self._save_activation(module_name, input[0])

      for name, module in model.named_modules():
         print(name)
         if isinstance(module, (nn.Conv2d, nn.Linear)): 
            hooks.append(
               module.register_forward_hook(partial( register_post_hook, name))
            )
            hooks.append(
               module.register_forward_pre_hook(register_pre_hook, name))

      return hooks

   def _save_activation(self, module_name, output):
      self.activations[module_name].append( output.detach().cpu().numpy())

# %%
