import logging

from typing import Any, Dict

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from torch import nn

logger: logging.Logger = logging.getLogger(__name__)


class LeffaTransform(nn.Module):
    def __init__(
        self,
        height: int = 1024,
        width: int = 768,
        dataset: str = "virtual_tryon",  # virtual_tryon or pose_transfer
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.dataset = dataset

        self.vae_processor = VaeImageProcessor(vae_scale_factor=8)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch_size = len(batch["src_image"])

        src_image_list = []
        ref_image_list = []
        mask_list = []
        densepose_list = []
        for i in range(batch_size):
            # 1. get original data
            src_image = batch["src_image"][i]
            ref_image = batch["ref_image"][i]
            mask = batch["mask"][i]
            densepose = batch["densepose"][i]

            # 3. process data
            src_image = self.vae_processor.preprocess(
                src_image, self.height, self.width)[0]
            ref_image = self.vae_processor.preprocess(
                ref_image, self.height, self.width)[0]
            mask = self.mask_processor.preprocess(
                mask, self.height, self.width)[0]
            if self.dataset in ["pose_transfer"]:
                densepose = densepose.resize(
                    (self.width, self.height), Image.NEAREST)
            else:
                densepose = self.vae_processor.preprocess(
                    densepose, self.height, self.width
                )[0]

            src_image = self.prepare_image(src_image)
            ref_image = self.prepare_image(ref_image)
            mask = self.prepare_mask(mask)
            if self.dataset in ["pose_transfer"]:
                densepose = self.prepare_densepose(densepose)
            else:
                densepose = self.prepare_image(densepose)

            src_image_list.append(src_image)
            ref_image_list.append(ref_image)
            mask_list.append(mask)
            densepose_list.append(densepose)

        src_image = torch.cat(src_image_list, dim=0)
        ref_image = torch.cat(ref_image_list, dim=0)
        mask = torch.cat(mask_list, dim=0)
        densepose = torch.cat(densepose_list, dim=0)

        batch["src_image"] = src_image
        batch["ref_image"] = ref_image
        batch["mask"] = mask
        batch["densepose"] = densepose

        return batch

    @staticmethod
    def prepare_image(image):
        if isinstance(image, torch.Tensor):
            # Batch single image
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = image.to(dtype=torch.float32)
        else:
            # preprocess image
            if isinstance(image, (Image.Image, np.ndarray)):
                image = [image]
            if isinstance(image, list) and isinstance(image[0], Image.Image):
                image = [np.array(i.convert("RGB"))[None, :] for i in image]
                image = np.concatenate(image, axis=0)
            elif isinstance(image, list) and isinstance(image[0], np.ndarray):
                image = np.concatenate([i[None, :] for i in image], axis=0)
            image = image.transpose(0, 3, 1, 2)
            image = torch.from_numpy(image).to(
                dtype=torch.float32) / 127.5 - 1.0
        return image

    @staticmethod
    def prepare_mask(mask):
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 2:
                # Batch and add channel dim for single mask
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[0] == 1:
                # Single mask, the 0'th dimension is considered to be
                # the existing batch size of 1
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[0] != 1:
                # Batch of mask, the 0'th dimension is considered to be
                # the batching dimension
                mask = mask.unsqueeze(1)

            # Binarize mask
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
        else:
            # preprocess mask
            if isinstance(mask, (Image.Image, np.ndarray)):
                mask = [mask]

            if isinstance(mask, list) and isinstance(mask[0], Image.Image):
                mask = np.concatenate(
                    [np.array(m.convert("L"))[None, None, :] for m in mask],
                    axis=0,
                )
                mask = mask.astype(np.float32) / 255.0
            elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
                mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            mask = torch.from_numpy(mask)

        return mask

    @staticmethod
    def prepare_densepose(densepose):
        """
        For internal (meta) densepose, the first and second channel should be normalized to 0~1 by 255.0,
        and the third channel should be normalized to 0~1 by 24.0
        """
        if isinstance(densepose, torch.Tensor):
            # Batch single densepose
            if densepose.ndim == 3:
                densepose = densepose.unsqueeze(0)
            densepose = densepose.to(dtype=torch.float32)
        else:
            # preprocess densepose
            if isinstance(densepose, (Image.Image, np.ndarray)):
                densepose = [densepose]
            if isinstance(densepose, list) and isinstance(
                densepose[0], Image.Image
            ):
                densepose = [np.array(i.convert("RGB"))[None, :]
                             for i in densepose]
                densepose = np.concatenate(densepose, axis=0)
            elif isinstance(densepose, list) and isinstance(densepose[0], np.ndarray):
                densepose = np.concatenate(
                    [i[None, :] for i in densepose], axis=0)
            densepose = densepose.transpose(0, 3, 1, 2)
            densepose = densepose.astype(np.float32)
            densepose[:, 0:2, :, :] /= 255.0
            densepose[:, 2:3, :, :] /= 24.0
            densepose = torch.from_numpy(densepose).to(
                dtype=torch.float32) * 2.0 - 1.0
        return densepose
