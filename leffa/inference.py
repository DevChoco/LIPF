from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from leffa.pipeline import LeffaPipeline


def pil_to_tensor(images):
    images = np.array(images).astype(np.float32) / 255.0
    images = torch.from_numpy(images.transpose(2, 0, 1))
    return images


class LeffaInference(object):
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model.to(self.device)
        self.model.eval()

        self.pipe = LeffaPipeline(model=self.model)

    def to_gpu(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
        return data

    def __call__(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        data = self.to_gpu(data)

        ref_acceleration = kwargs.get("ref_acceleration", False)
        num_inference_steps = kwargs.get("num_inference_steps", 50)
        guidance_scale = kwargs.get("guidance_scale", 2.5)
        seed = kwargs.get("seed", 42)
        repaint = kwargs.get("repaint", False)
        generator = torch.Generator(self.pipe.device).manual_seed(seed)
        
        # Extract prompt and negative_prompt if provided
        prompt = kwargs.get("prompt", None)
        negative_prompt = kwargs.get("negative_prompt", None)
        
        images = self.pipe(
            src_image=data["src_image"],
            ref_image=data["ref_image"],
            mask=data["mask"],
            densepose=data["densepose"],
            ref_acceleration=ref_acceleration,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            repaint=repaint,
            # Only pass if not None
            prompt=prompt,
            negative_prompt=negative_prompt
        )[0]

        # images = [pil_to_tensor(image) for image in images]
        # images = torch.stack(images)

        outputs = {}
        outputs["src_image"] = (data["src_image"] + 1.0) / 2.0
        outputs["ref_image"] = (data["ref_image"] + 1.0) / 2.0
        outputs["generated_image"] = images
        return outputs
