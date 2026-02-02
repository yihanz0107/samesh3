import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from omegaconf import OmegaConf
from transformers import pipeline
from data.common import NumpyTensor

class Sam3Model(nn.Module):
    """
    SAM3 automatic mask-generation using HuggingFace transformers pipeline.

    Call: masks = model(PIL.Image) -> np.ndarray (n,h,w) bool
    """
    def __init__(self, config: OmegaConf, device="cuda:0"):
        super().__init__()
        self.config = config
        self.device_str = device


        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            self.pipe_device = 0
        elif isinstance(device, int) and device >= 0 and torch.cuda.is_available():
            self.pipe_device = device
        else:
            self.pipe_device = -1

        model_id = config.sam.get("model", None) or config.sam.get("checkpoint", None) or config.sam.get("model_dir", None)
        if model_id is None:
            raise ValueError("config.sam.model / checkpoint / model_dir 需要提供一个（本地目录或模型名）")

        self.pipe = pipeline(
            task="mask-generation",
            model=model_id,
            device=self.pipe_device,
        )

        # Optional: Limit the maximum number of masks per image (to prevent image overload).
        self.max_masks = int(config.sam.get("max_masks", 0))  # 0 means no maximum

    def forward(self, image: Image.Image) -> NumpyTensor['n h w']:
        with torch.no_grad():
            out = self.pipe(image)

        masks = out["masks"]  # list[torch.Tensor(H,W)]
        if self.max_masks > 0:
            masks = masks[: self.max_masks]

        bmasks = np.stack([m.to("cpu").numpy().astype(bool) for m in masks], axis=0)
        return bmasks



def combine_bmasks(masks: NumpyTensor['n h w'], sort=False) -> NumpyTensor['h w']:
    """
    """
    mask_combined = np.zeros_like(masks[0], dtype=int)
    if sort:
        masks = sorted(masks, key=lambda x: x.sum(), reverse=True)
    for i, mask in enumerate(masks):
        mask_combined[mask] = i + 1
    return mask_combined


def remove_artifacts(mask: NumpyTensor['h w'], mode: str, min_area=128) -> NumpyTensor['h w']:
    """
    Removes small islands/fill holes from a mask.
    """
    assert mode in ['holes', 'islands']
    mode_holes = (mode == 'holes')

    def remove_helper(bmask):
        # opencv connected components operates on binary masks only
        bmask = (mode_holes ^ bmask).astype(np.uint8)
        nregions, regions, stats, _ = cv2.connectedComponentsWithStats(bmask, 8)
        sizes = stats[:, -1][1:]  # Row 0 corresponds to 0 pixels
        fill = [i + 1 for i, s in enumerate(sizes) if s < min_area] + [0]
        if not mode_holes:
            fill = [i for i in range(nregions) if i not in fill]
        return np.isin(regions, fill)

    mask_combined = np.zeros_like(mask)
    for label in np.unique(mask): # also process background
        mask_combined[remove_helper(mask == label)] = label
    return mask_combined


def colormap_mask(
    mask : NumpyTensor['h w'], 
    image: NumpyTensor['h w 3']=None, background=np.array([255, 255, 255]), foreground=None, blend=0.25
) -> Image.Image:
    """
    """
    palette = np.random.randint(0, 255, (np.max(mask) + 1, 3))
    palette[0] = background
    if foreground is not None:
        for i in range(1, len(palette)):
            palette[i] = foreground
    image_mask = palette[mask.astype(int)] # type conversion for boolean masks
    image_blend = image_mask if image is None else image_mask * (1 - blend) + image * blend
    image_blend = np.clip(image_blend, 0, 255).astype(np.uint8)
    return Image.fromarray(image_blend)


