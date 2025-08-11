import numpy as np
from PIL import Image
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
import cv2 
import os

class LeffaVirtualTryOn:
    def __init__(self, ckpt_dir: str):
        self.mask_predictor = AutoMasker(
            densepose_path=f"{ckpt_dir}/densepose",
            schp_path=f"{ckpt_dir}/schp",
        )
        self.densepose_predictor = DensePosePredictor(
            config_path=f"{ckpt_dir}/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path=f"{ckpt_dir}/densepose/model_final_162be9.pkl",
        )
        self.parsing = Parsing(
            atr_path=f"{ckpt_dir}/humanparsing/parsing_atr.onnx",
            lip_path=f"{ckpt_dir}/humanparsing/parsing_lip.onnx",
        )
        self.openpose = OpenPose(
            body_model_path=f"{ckpt_dir}/openpose/body_pose_model.pth",
        )

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path=f"{ckpt_dir}/stable-diffusion-inpainting",
            pretrained_model=f"{ckpt_dir}/virtual_tryon.pth",
            dtype="float16",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

        self.skin_model = LeffaModel(
            pretrained_model_name_or_path=f"{ckpt_dir}/stable-diffusion-inpainting",
            pretrained_model=f"{ckpt_dir}/virtual_tryon.pth",
            dtype="float16",
        )
        self.skin_inference = LeffaInference(model=self.skin_model)

        vt_model_dc = LeffaModel(
            pretrained_model_name_or_path=f"{ckpt_dir}/stable-diffusion-inpainting",
            pretrained_model=f"{ckpt_dir}/virtual_tryon_dc.pth",
            dtype="float16",
        )
        self.vt_inference_dc = LeffaInference(model=vt_model_hd)

        # majicmix realistic skin model - diffusers pipeline
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
        )
        self.skin_pipe = StableDiffusionControlNetInpaintPipeline.from_single_file(
            f"{ckpt_dir}/majicmixRealistic_v7.safetensors",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")

    def generate_skin(
        self,
        src_image: Image.Image,
        inpaint_mask_img: Image.Image,
        step: int = 20,
        seed: int = 42
    ) -> Image.Image:
        """
        주어진 마스크 영역에 사실적인 피부를 인페인팅합니다.
        Inpaints realistic skin in the given masked area using a dedicated skin model.
        """
        
        skin_prompt = "Wearing Held Tight Short Sleeve Shirt, high quality skin, realistic, high quality"
        negative_prompt = "Blurry, low quality, artifacts, deformed, ugly, , texture, watermark, text, bad anatomy, extra limbs, face, hands, fingers"

        openpose_result = self.openpose(src_image)
        
        if isinstance(openpose_result, dict):
            openpose_image = openpose_result.get("image")
        else:
            openpose_image = openpose_result
        
        if not isinstance(openpose_image, Image.Image):
            raise TypeError(f"OpenPose에서 반환된 control image가 유효하지 않습니다: {type(openpose_image)}")
        
        generator = torch.Generator(device="cuda").manual_seed(seed)

        generated_image = self.skin_pipe(
            prompt=skin_prompt,
            negative_prompt=negative_prompt,
            image=src_image,
            mask_image=inpaint_mask_img,
            control_image=openpose_image,
            width=src_image.width,
            height=src_image.height,
            num_inference_steps=step,
            generator=generator,
            guidance_scale=7.0  # Lower guidance to better match image context
        ).images[0]

        src_np = np.array(src_image)
        mask_np = np.array(inpaint_mask_img.convert("L")) / 255.0
        mask_np = np.expand_dims(mask_np, axis=-1)
        generated_np = np.array(generated_image)

        final_np = src_np * (1 - mask_np) + generated_np * mask_np
        final_image = Image.fromarray(final_np.astype(np.uint8))

        return final_image

    def skin_test(
        self,
        src_image_path: str,
        ref_image_path: str, 
        control_type: str,
        output_path: str = None,
        step: int = 40,
        seed: int = 42,
        cross_attention_kwargs={"scale": 3},
        vt_model_type: str = "viton_hd",
        vt_garment_type: str = "upper_body",
        vt_repaint: bool = False,
        ref_acceleration: bool = False,
        src_mask_path: str = None     
    ):
        """
        이미지에서 팔과 다리의 옷을 제거하고 사실적인 피부로 인페인팅합니다.
        Removes clothing from arms and legs in an image and inpaints with realistic skin.
        """
        src_image = Image.open(src_image_path).convert("RGB")
        src_image = resize_and_center(src_image, 768, 1024)

        garment_mask_img = self.mask_predictor(src_image, "overall")["mask"]
        garment_mask_np = np.array(garment_mask_img.convert("L")) > 128

        parsing_map, _ = self.parsing(src_image.resize((768, 1024)))
        parsing_map = np.array(parsing_map)
        limb_mask_raw = np.isin(parsing_map, [4, 5]).astype(np.uint8)
        limb_mask_img = Image.fromarray(limb_mask_raw * 255).resize(src_image.size, Image.NEAREST)
        limb_mask_np = np.array(limb_mask_img.convert("L")) > 128
        
        if vt_garment_type == "upper_body":
            garment_mask_np = np.isin(parsing_map, [4]).astype(np.uint8)
            hands_mask_np = np.isin(parsing_map, [14, 15]).astype(np.uint8) 
            garment_mask_np |= hands_mask_np
        elif vt_garment_type == "lower_body":
            garment_mask_np = np.isin(parsing_map, [5]).astype(np.uint8)
        else:
            garment_mask_np = np.isin(parsing_map, [4, 5]).astype(np.uint8)

        inpaint_mask_np = garment_mask_np & limb_mask_np
        
        kernel = np.ones((10, 10), np.uint8)  # 10px 마진용 커널
        inpaint_mask_np_dilated = cv2.dilate(inpaint_mask_np.astype(np.uint8), kernel, iterations=1)
        
        inpaint_mask_img = Image.fromarray(inpaint_mask_np_dilated * 255)

        if not np.any(inpaint_mask_np):
            if output_path:
                src_image.save(output_path)
            return src_image, Image.fromarray(np.zeros_like(inpaint_mask_np, dtype=np.uint8) * 255)

        final_image = self.generate_skin(
            src_image=src_image,
            inpaint_mask_img=inpaint_mask_img,
            step=step,
            seed=seed
        )

        agnostic_image = final_image

        return final_image, inpaint_mask_img, parsing_map

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        output_path: str = None,
        step=20,
        cross_attention_kwargs={"scale": 3},
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        src_mask_path=None
    ):
        assert control_type in ["virtual_tryon", "pose_transfer"], f"Invalid control type: {control_type}"

        src_image = Image.open(src_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        garment_mask_img = self.mask_predictor(src_image, "overall")["mask"]
        garment_mask_np = np.array(garment_mask_img.convert("L")) > 128

        parsing_map, _ = self.parsing(src_image.resize((768, 1024)))
        parsing_map = np.array(parsing_map)
        limb_mask_raw = np.isin(parsing_map, [4, 5]).astype(np.uint8)
        limb_mask_img = Image.fromarray(limb_mask_raw * 255).resize(src_image.size, Image.NEAREST)
        limb_mask_np = np.array(limb_mask_img.convert("L")) > 128

        if vt_garment_type == "upper_body":
            garment_mask_np = np.isin(parsing_map, [4]).astype(np.uint8)
            hands_mask_np = np.isin(parsing_map, [14, 15]).astype(np.uint8)
            garment_mask_np |= hands_mask_np
        elif vt_garment_type == "lower_body":
            garment_mask_np = np.isin(parsing_map, [5]).astype(np.uint8)
        else:
            garment_mask_np = np.isin(parsing_map, [4, 5]).astype(np.uint8)

        inpaint_mask_np = garment_mask_np & limb_mask_np

        kernel = np.ones((10, 10), np.uint8)  # 10px 마진용 커널
        inpaint_mask_np_dilated = cv2.dilate(inpaint_mask_np.astype(np.uint8), kernel, iterations=1)

        inpaint_mask_img = Image.fromarray(inpaint_mask_np_dilated * 255)

        if not np.any(inpaint_mask_np):
            if output_path:
                src_image.save(output_path)
            return src_image, Image.fromarray(np.zeros_like(inpaint_mask_np, dtype=np.uint8) * 255)

        final_image = self.generate_skin(
            src_image=src_image,
            inpaint_mask_img=inpaint_mask_img,
            step=step,
            seed=seed
        )
        
        agnostic_image = final_image
        
        if control_type == "virtual_tryon":
            garment_mapping = {
                "dresses": "overall",
                "upper_body": "upper",
                "lower_body": "lower",
                "short_sleeve": "short_sleeve",
                "shorts": "shorts"
            }
            garment_type_hd = garment_mapping.get(vt_garment_type, "upper")
            mask = self.mask_predictor(agnostic_image, garment_type_hd)["mask"]
            
            if src_mask_path:
                mask.save(src_mask_path)
        else:
            mask = Image.fromarray(np.ones_like(agnostic_np, dtype=np.uint8) * 255)
    
        agnostic_np = np.array(agnostic_image)
        if vt_model_type == "viton_hd":
            seg = self.densepose_predictor.predict_seg(agnostic_np)[:, :, ::-1]
        else:
            iuv = self.densepose_predictor.predict_iuv(agnostic_np)
            seg = np.concatenate([iuv[:, :, :1]] * 3, axis=-1)
        densepose = Image.fromarray(seg)
    
        transform = LeffaTransform()
        data = {
            "src_image": [agnostic_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
    
        inference = self.vt_inference_hd if vt_model_type == "viton_hd" else self.vt_inference_dc
        
        garment_prompt = "High quality skin, lifelike details, realistic textures, full masking range"
        negative_prompt = "distorted, blurry, low quality, artifact, background, clothes"
        
        result = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            cross_attention_kwargs=cross_attention_kwargs,
            seed=seed,
            repaint=vt_repaint,
            prompt=garment_prompt,
            negative_prompt=negative_prompt
        )
    
        gen_image = result["generated_image"][0]
    
        if output_path:
            gen_image.save(output_path)

            gt_path = os.path.join(os.path.dirname(output_path), "ground_truth.png")
            pred_path = os.path.join(os.path.dirname(output_path), "prediction.png")

            ref_image.save(gt_path)
            gen_image.save(pred_path)

            print(f"Ground Truth saved at: {gt_path}")
            print(f"Prediction saved at: {pred_path}")

        return gen_image, mask, densepose, agnostic_image