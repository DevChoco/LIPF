import os
from typing import Union

import cv2
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from SCHP import SCHP  # type: ignore

from leffa_utils.densepose_for_mask import DensePose  # type: ignore

DENSE_INDEX_MAP = {
    "background": [0],
    "torso": [1, 2],
    "right hand": [3],
    "left hand": [4],
    "right foot": [5],
    "left foot": [6],
    "right thigh": [7, 9],
    "left thigh": [8, 10],
    "right leg": [11, 13],
    "left leg": [12, 14],
    "left big arm": [15, 17],
    "right big arm": [16, 18],
    "left forearm": [19, 21],
    "right forearm": [20, 22],
    "face": [23, 24],
    "thighs": [7, 8, 9, 10],
    "legs": [11, 12, 13, 14],
    "hands": [3, 4],
    "feet": [5, 6],
    "big arms": [15, 16, 17, 18],
    "forearms": [19, 20, 21, 22],
}

ATR_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Sunglasses": 3,
    "Upper-clothes": 4,
    "Skirt": 5,
    "Pants": 6,
    "Dress": 7,
    "Belt": 8,
    "Left-shoe": 9,
    "Right-shoe": 10,
    "Face": 11,
    "Left-leg": 12,
    "Right-leg": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Bag": 16,
    "Scarf": 17,
}

LIP_MAPPING = {
    "Background": 0,
    "Hat": 1,
    "Hair": 2,
    "Glove": 3,
    "Sunglasses": 4,
    "Upper-clothes": 5,
    "Dress": 6,
    "Coat": 7,
    "Socks": 8,
    "Pants": 9,
    "Jumpsuits": 10,
    "Scarf": 11,
    "Skirt": 12,
    "Face": 13,
    "Left-arm": 14,
    "Right-arm": 15,
    "Left-leg": 16,
    "Right-leg": 17,
    "Left-shoe": 18,
    "Right-shoe": 19,
}

PROTECT_BODY_PARTS = {
    "upper": ["Left-leg", "Right-leg"],
    "lower": ["Right-arm", "Left-arm", "Face"],
    "overall": [],
    "inner": ["Left-leg", "Right-leg"],
    "outer": ["Left-leg", "Right-leg"],
    "short_sleeve": ["Left-leg", "Right-leg", "Face"],
    "shorts": ["Right-arm", "Left-arm", "Face"],
}
PROTECT_CLOTH_PARTS = {
    "upper": {
        "ATR": ["Skirt", "Pants"],
        "LIP": ["Skirt", "Pants"]
    },
    "lower": {
        "ATR": ["Upper-clothes"],
        "LIP": ["Upper-clothes", "Coat"]
    },
    "overall": {"ATR": [], "LIP": []},
    "inner": {
        "ATR": ["Dress", "Coat", "Skirt", "Pants"],
        "LIP": ["Dress", "Coat", "Skirt", "Pants", "Jumpsuits"],
    },
    "outer": {
        "ATR": ["Dress", "Pants", "Skirt"],
        "LIP": ["Upper-clothes", "Dress", "Pants", "Skirt", "Jumpsuits"],
    },
    "short_sleeve": {
        "ATR": ["Pants", "Skirt", "Dress"],
        "LIP": ["Pants", "Skirt", "Dress"],
    },
    "shorts": {
        "ATR": ["Upper-clothes", "Coat", "Dress"],
        "LIP": ["Upper-clothes", "Coat", "Dress"],
    },
}
MASK_CLOTH_PARTS = {
    "upper": ["Upper-clothes", "Coat", "Dress", "Jumpsuits"],
    "lower": ["Pants", "Skirt", "Dress", "Jumpsuits"],
    "overall": ["Upper-clothes", "Dress", "Pants", "Skirt", "Coat", "Jumpsuits"],
    "inner": ["Upper-clothes"],
    "outer": ["Coat"],
    "short_sleeve": ["Upper-clothes", "Coat"],
    "shorts": ["Pants", "Skirt"],
}

# Fixed: Forearms removed from short_sleeve masking
MASK_DENSE_PARTS = {
    "upper": ["torso", "big arms", "forearms"],
    "lower": ["thighs", "legs"],
    "overall": ["torso", "thighs", "legs", "big arms", "forearms"],
    "inner": ["torso"],
    "outer": ["torso", "big arms", "forearms"],
    "short_sleeve": ["torso", "big arms"],  # Forearms removed for short sleeves
    "shorts": ["thighs"],
}

schp_public_protect_parts = [
    "Hat",
    "Hair",
    "Sunglasses",
    "Left-shoe",
    "Right-shoe",
    "Bag",
    "Glove",
    "Scarf",
]
schp_protect_parts = {
    "upper": ["Left-leg", "Right-leg", "Skirt", "Pants", "Jumpsuits"],
    "lower": ["Left-arm", "Right-arm", "Upper-clothes", "Coat"],
    "overall": [],
    "inner": ["Left-leg", "Right-leg", "Skirt", "Pants", "Jumpsuits", "Coat"],
    "outer": ["Left-leg", "Right-leg", "Skirt", "Pants", "Jumpsuits", "Upper-clothes"],
    "short_sleeve": ["Left-arm", "Left-leg", "Right-leg", "Skirt", "Pants", "Jumpsuits"],
    "shorts": ["Left-arm", "Right-arm", "Upper-clothes", "Coat", "Dress"],  # 다리 보호 없음
}
schp_mask_parts = {
    "upper": ["Upper-clothes", "Dress", "Coat", "Jumpsuits"],
    "lower": ["Pants", "Skirt", "Dress", "Jumpsuits", "socks"],
    "overall": [
        "Upper-clothes",
        "Dress",
        "Pants",
        "Skirt",
        "Coat",
        "Jumpsuits",
        "socks",
    ],
    "inner": ["Upper-clothes"],
    "outer": ["Coat"],
    "short_sleeve": ["Upper-clothes", "Coat"],
    "shorts": ["Pants", "Skirt", "Jumpsuits"],
}

# Fixed: Forearms removed from short_sleeve masking
dense_mask_parts = MASK_DENSE_PARTS.copy()


def vis_mask(image, mask):
    image = np.array(image).astype(np.uint8)
    mask = np.array(mask).astype(np.uint8)
    mask[mask > 127] = 255
    mask[mask <= 127] = 0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.repeat(mask, 3, axis=-1)
    mask = mask / 255
    return Image.fromarray((image * (1 - mask)).astype(np.uint8))


def part_mask_of(part: Union[str, list], parse: np.ndarray, mapping: dict):
    if isinstance(part, str):
        part = [part]
    mask = np.zeros_like(parse)
    for _ in part:
        if _ not in mapping:
            continue
        if isinstance(mapping[_], list):
            for i in mapping[_]:
                mask += parse == i
        else:
            mask += parse == mapping[_]
    return mask

def hull_mask(mask_area: np.ndarray):
    ret, binary = cv2.threshold(mask_area, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull_mask = np.zeros_like(mask_area)
    for c in contours:
        hull = cv2.convexHull(c)
        hull_mask = cv2.fillPoly(np.zeros_like(mask_area), [hull], 255) | hull_mask
    return hull_mask


class AutoMasker:
    def __init__(
        self,
        densepose_path: str = "./ckpts/densepose",
        schp_path: str = "./ckpts/schp",
        device="cuda",
    ):
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        self.densepose_processor = DensePose(densepose_path, device)
        self.schp_processor_atr = SCHP(
            ckpt_path=os.path.join(schp_path, "exp-schp-201908301523-atr.pth"),
            device=device,
        )
        self.schp_processor_lip = SCHP(
            ckpt_path=os.path.join(schp_path, "exp-schp-201908261155-lip.pth"),
            device=device,
        )

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

    def process_densepose(self, image_or_path):
        return self.densepose_processor(image_or_path, resize=1024)

    def process_schp_lip(self, image_or_path):
        return self.schp_processor_lip(image_or_path)

    def process_schp_atr(self, image_or_path):
        return self.schp_processor_atr(image_or_path)

    def preprocess_image(self, image_or_path):
        return {
            "densepose": self.densepose_processor(image_or_path, resize=1024),
            "schp_atr": self.schp_processor_atr(image_or_path),
            "schp_lip": self.schp_processor_lip(image_or_path),
        }

    @staticmethod
    def cloth_agnostic_mask(
        densepose_mask: Image.Image,
        schp_lip_mask: Image.Image,
        schp_atr_mask: Image.Image,
        part: str = "overall",
        **kwargs,
    ):
        assert part in ["upper", "lower", "overall", "inner", "outer", "short_sleeve", "shorts"], f"Invalid part: {part}"
        w, h = densepose_mask.size

        dilate_kernel = max(w, h) // 250
        dilate_kernel = dilate_kernel if dilate_kernel % 2 == 1 else dilate_kernel + 1
        dilate_kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)

        kernal_size = max(w, h) // 25
        kernal_size = kernal_size if kernal_size % 2 == 1 else kernal_size + 1

        densepose_mask = np.array(densepose_mask)
        schp_lip_mask = np.array(schp_lip_mask)
        schp_atr_mask = np.array(schp_atr_mask)

        # Strong Protect Area (Hands, Face, Accessory, Feet)
        hands_protect_area = part_mask_of(["hands", "feet"], densepose_mask, DENSE_INDEX_MAP)
        hands_protect_area = cv2.dilate(hands_protect_area, dilate_kernel, iterations=1)
        hands_protect_area = hands_protect_area & (
            part_mask_of(["Left-arm", "Right-arm", "Left-leg", "Right-leg"], schp_atr_mask, ATR_MAPPING) |
            part_mask_of(["Left-arm", "Right-arm", "Left-leg", "Right-leg"], schp_lip_mask, LIP_MAPPING)
        )
        face_protect_area = part_mask_of("Face", schp_lip_mask, LIP_MAPPING)

        strong_protect_area = hands_protect_area | face_protect_area
        if part == "short_sleeve":
            forearm_protect = part_mask_of(["forearms"], densepose_mask, DENSE_INDEX_MAP)
            forearm_protect = cv2.dilate(forearm_protect, dilate_kernel, iterations=1)
            strong_protect_area |= forearm_protect
        if part == "upper":
            hands_area = part_mask_of(["hands"], densepose_mask, DENSE_INDEX_MAP)
            hands_area = cv2.dilate(hands_area, dilate_kernel, iterations=1)
            strong_protect_area |= hands_area

        # Weak Protect Area (Hair, Irrelevant Clothes, Body Parts)
        body_protect_area = part_mask_of(
            PROTECT_BODY_PARTS[part], schp_lip_mask, LIP_MAPPING
        ) | part_mask_of(PROTECT_BODY_PARTS[part], schp_atr_mask, ATR_MAPPING)
        hair_protect_area = part_mask_of(
            ["Hair"], schp_lip_mask, LIP_MAPPING
        ) | part_mask_of(["Hair"], schp_atr_mask, ATR_MAPPING)
        cloth_protect_area = part_mask_of(
            PROTECT_CLOTH_PARTS[part]["LIP"], schp_lip_mask, LIP_MAPPING
        ) | part_mask_of(PROTECT_CLOTH_PARTS[part]["ATR"], schp_atr_mask, ATR_MAPPING)
        accessory_protect_area = part_mask_of(
            (
                accessory_parts := [
                    "Hat",
                    "Glove",
                    "Sunglasses",
                    "Bag",
                    "Left-shoe",
                    "Right-shoe",
                    "Scarf",
                    "Socks",
                ]
            ),
            schp_lip_mask,
            LIP_MAPPING,
        ) | part_mask_of(accessory_parts, schp_atr_mask, ATR_MAPPING)
        weak_protect_area = (
            body_protect_area
            | cloth_protect_area
            | hair_protect_area
            | strong_protect_area
            | accessory_protect_area
        )

        # Mask Area
        strong_mask_area = part_mask_of(
            MASK_CLOTH_PARTS[part], schp_lip_mask, LIP_MAPPING
        ) | part_mask_of(MASK_CLOTH_PARTS[part], schp_atr_mask, ATR_MAPPING)
        background_area = part_mask_of(
            ["Background"], schp_lip_mask, LIP_MAPPING
        ) & part_mask_of(["Background"], schp_atr_mask, ATR_MAPPING)
        mask_dense_area = part_mask_of(
            MASK_DENSE_PARTS[part], densepose_mask, DENSE_INDEX_MAP
        )
        mask_dense_area = cv2.resize(
            mask_dense_area.astype(np.uint8),
            None,
            fx=0.25,
            fy=0.25,
            interpolation=cv2.INTER_NEAREST,
        )
        mask_dense_area = cv2.dilate(mask_dense_area, dilate_kernel, iterations=2)
        mask_dense_area = cv2.resize(
            mask_dense_area.astype(np.uint8),
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_NEAREST,
        )

        mask_area = (
            np.ones_like(densepose_mask) & (~weak_protect_area) & (~background_area)
        ) | mask_dense_area

        mask_area = (
            hull_mask(mask_area * 255) // 255
        )  # Convex Hull to expand the mask area
        mask_area = mask_area & (~weak_protect_area)
        mask_area = cv2.GaussianBlur(mask_area * 255, (kernal_size, kernal_size), 0)
        mask_area[mask_area < 25] = 0
        mask_area[mask_area >= 25] = 1
        mask_area = (mask_area | strong_mask_area) & (~strong_protect_area)
        mask_area = cv2.dilate(mask_area, dilate_kernel, iterations=1)

        return Image.fromarray(mask_area * 255)

    def __call__(
        self,
        image: Union[str, Image.Image],
        mask_type: str = "upper",
    ):
        assert mask_type in [
            "upper",
            "lower",
            "overall",
            "inner",
            "outer",
            "short_sleeve",
            "shorts",
        ], f"mask_type should be one of ['upper', 'lower', 'overall', 'inner', 'outer', 'short_sleeve', 'shorts'], but got {mask_type}"
        preprocess_results = self.preprocess_image(image)
        mask = self.cloth_agnostic_mask(
            preprocess_results["densepose"],
            preprocess_results["schp_lip"],
            preprocess_results["schp_atr"],
            part=mask_type,
        )
        return {
            "mask": mask,
            "densepose": preprocess_results["densepose"],
            "schp_lip": preprocess_results["schp_lip"],
            "schp_atr": preprocess_results["schp_atr"],
        }


if __name__ == "__main__":
    import os
    import sys

    from PIL import Image

    automasker = AutoMasker()

    image_path = sys.argv[1]
    image = Image.open(image_path).convert("RGB")
    outputs = automasker(
        image,
        "upper",
        # "lower",
    )
    mask = outputs["mask"]
    # densepose = outputs["densepose"]  # densepose I map, range 0~24
    # schp_lip = outputs["schp_lip"]
    # schp_atr = outputs["schp_atr"]
    mask.save(".".join(image_path.split(".")[:-1]) + "_mask.jpg")
