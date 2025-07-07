# import numpy as np
# from PIL import Image
# from leffa.transform import LeffaTransform
# from leffa.model import LeffaModel
# from leffa.inference import LeffaInference
# from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
# from leffa_utils.densepose_predictor import DensePosePredictor
# from leffa_utils.utils import resize_and_center, get_agnostic_mask_hd, get_agnostic_mask_dc, preprocess_garment_image
# from preprocess.humanparsing.run_parsing import Parsing
# from preprocess.openpose.run_openpose import OpenPose
# import torch

# class LeffaVirtualTryOn:
#     def __init__(self, ckpt_dir: str):
#         self.mask_predictor = AutoMasker(
#             densepose_path=f"{ckpt_dir}/densepose",
#             schp_path=f"{ckpt_dir}/schp",
#         )
#         self.densepose_predictor = DensePosePredictor(
#             config_path=f"{ckpt_dir}/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
#             weights_path=f"{ckpt_dir}/densepose/model_final_162be9.pkl",
#         )
#         self.parsing = Parsing(
#             atr_path=f"{ckpt_dir}/humanparsing/parsing_atr.onnx",
#             lip_path=f"{ckpt_dir}/humanparsing/parsing_lip.onnx",
#         )
#         self.openpose = OpenPose(
#             body_model_path=f"{ckpt_dir}/openpose/body_pose_model.pth",
#         )
        
#         vt_model_hd = LeffaModel(
#             pretrained_model_name_or_path=f"{ckpt_dir}/stable-diffusion-inpainting",
#             pretrained_model=f"{ckpt_dir}/virtual_tryon.pth",
#             dtype="float16",
#         )
#         self.vt_inference_hd = LeffaInference(model=vt_model_hd)

#         self.skin_model = LeffaModel(
#             pretrained_model_name_or_path=f"{ckpt_dir}/stable-diffusion-inpainting",
#             pretrained_model=f"{ckpt_dir}/virtual_tryon.pth",  # 기존 모델 재사용
#             dtype="float16",
#         )
#         self.skin_inference = LeffaInference(model=self.skin_model)

#         vt_model_dc = LeffaModel(
#             pretrained_model_name_or_path=f"{ckpt_dir}/stable-diffusion-inpainting",
#             pretrained_model=f"{ckpt_dir}/virtual_tryon_dc.pth",
#             dtype="float16",
#         )
#         self.vt_inference_dc = LeffaInference(model=vt_model_hd)

#         # majicMIX model with safetensors
#         self.majic_model = LeffaModel(
#             pretrained_model_name_or_path=f"{ckpt_dir}/stable-diffusion-inpainting",
#             pretrained_model=f"{ckpt_dir}/majicmixRealistic_v7.safetensors",  # .safetensors file
#             dtype="float16",
#         )
#         self.majic_inference = LeffaInference(model=self.majic_model)

#     def generate_skin(self, image: Image.Image, mask: Image.Image, vt_model_type: str):
#         try:
#             """Generate skin using inpainting model"""
#             prompt = "Realistic female human skin texture, natural skin tone, nude, high quality, detailed skin with realistic texture"
#             negative_prompt = "background, scenery, clothes, deformed, deformed, blurry, low quality, artifacts"
            
#             src_image_array = np.array(image)
            
#             # DensePose 생성 (vt_model_type 사용)
#             if vt_model_type == "viton_hd":
#                 seg = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
#             else:
#                 iuv = self.densepose_predictor.predict_iuv(src_image_array)
#                 seg = np.concatenate([iuv[:, :, :1]] * 3, axis=-1)
#             densepose = Image.fromarray(seg)
            
#             # 데이터 준비
#             transform = LeffaTransform()
#             data = {
#                 "src_image": [image],
#                 "ref_image": [image],
#                 "mask": [mask],
#                 "densepose": [densepose],
#             }
#             data = transform(data)
            
#             # 피부 생성 실행
#             result = self.skin_inference(
#                 data,
#                 ref_acceleration=False,
#                 prompt=prompt,
#                 negative_prompt=negative_prompt,
#                 num_inference_steps=25,  # 단계 줄임
#                 guidance_scale=6.5,      # 스케일 줄임
#                 repaint=False
#             )
#             return result["generated_image"][0]
        
#         except Exception as e:
#             print(f"❌ Skin generation failed: {e}")
#             print("🔄 Using white background fallback")
#             # 실패시 흰색 배경 반환
#             return Image.new("RGB", image.size, (255, 255, 255))

#     def leffa_predict(
#         self,
#         src_image_path,
#         ref_image_path,
#         control_type,
#         ref_acceleration=False,
#         output_path: str = None,
#         step=30,
#         cross_attention_kwargs={"scale": 3},
#         seed=42,
#         vt_model_type="viton_hd",
#         vt_garment_type="upper_body",
#         vt_repaint=False,
#         src_mask_path=None
#     ):
#         assert control_type in ["virtual_tryon", "pose_transfer"], f"Invalid control type: {control_type}"
        
#         # 1. 이미지 로드 및 리사이징
#         src_image = Image.open(src_image_path).convert("RGB")
#         ref_image = Image.open(ref_image_path).convert("RGB")
#         src_image = resize_and_center(src_image, 768, 1024)
#         ref_image = resize_and_center(ref_image, 768, 1024)
        
#         # 2. 전신 마스크 생성 (옷 제거용)
#         full_body_mask = self.mask_predictor(src_image, "overall")["mask"]
        
#         # 3. 피부 생성용 마스크 준비 (반전 제거)
#         skin_mask = full_body_mask
        
#         # 4. 디퓨전으로 피부 생성 (vt_model_type 전달)
#         skin_image = self.generate_skin(src_image, skin_mask, vt_model_type)
        
#         # 5. 옷 제거 및 피부 적용
#         src_np = np.array(src_image)
#         mask_np = np.array(full_body_mask.convert("L")) / 255.0
#         mask_np = np.expand_dims(mask_np, axis=2)
#         skin_np = np.array(skin_image)
        
#         # 수정된 블렌딩: 피사체 영역에 생성된 피부 적용
#         agnostic_np = src_np * (1 - mask_np) + skin_np * mask_np
#         agnostic_image = Image.fromarray(agnostic_np.astype(np.uint8))
        
#         # 6. 메인 마스크 생성
#         if control_type == "virtual_tryon":
#             garment_mapping = {
#                 "dresses": "overall",
#                 "upper_body": "upper",
#                 "lower_body": "lower",
#                 "short_sleeve": "short_sleeve",
#                 "shorts": "shorts"
#             }
#             garment_type_hd = garment_mapping.get(vt_garment_type, "upper")
#             mask = self.mask_predictor(agnostic_image, garment_type_hd)["mask"]
            
#             if src_mask_path:
#                 mask.save(src_mask_path)
#         else:
#             mask = Image.fromarray(np.ones_like(agnostic_np, dtype=np.uint8) * 255)
    
#         # 7. DensePose 생성
#         agnostic_np = np.array(agnostic_image)
#         if vt_model_type == "viton_hd":
#             seg = self.densepose_predictor.predict_seg(agnostic_np)[:, :, ::-1]
#         else:
#             iuv = self.densepose_predictor.predict_iuv(agnostic_np)
#             seg = np.concatenate([iuv[:, :, :1]] * 3, axis=-1)
#         densepose = Image.fromarray(seg)
    
#         # 8. 최종 가상 피팅
#         transform = LeffaTransform()
#         data = {
#             "src_image": [agnostic_image],
#             "ref_image": [ref_image],
#             "mask": [mask],
#             "densepose": [densepose],
#         }
#         data = transform(data)
    
#         inference = self.vt_inference_hd if vt_model_type == "viton_hd" else self.vt_inference_dc
        
#         garment_prompt = "high quality clothing, detailed fabric, realistic texture"
#         negative_prompt = "deformed, blurry, low quality, artifacts"
        
#         result = inference(
#             data,
#             ref_acceleration=ref_acceleration,
#             num_inference_steps=step,
#             cross_attention_kwargs={"scale": cross_attention_kwargs},
#             seed=seed,
#             repaint=vt_repaint,
#             prompt=garment_prompt,
#             negative_prompt=negative_prompt
#         )
    
#         gen_image = result["generated_image"][0]
    
#         # 9. 결과 반환
#         if output_path:
#             gen_image.save(output_path)
        
#         return gen_image, mask, densepose, full_body_mask, agnostic_image, skin_image





















# vton_script.py

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
        step: int = 40,
        seed: int = 42
    ) -> Image.Image:
        """
        주어진 마스크 영역에 사실적인 피부를 인페인팅합니다.
        Inpaints realistic skin in the given masked area using a dedicated skin model.
        """
        # To generate skin that matches the person, use a more neutral prompt
        # and guide the model to be less creative.
        skin_prompt = "human skin, realistic, high quality"
        negative_prompt = "blurry, low quality, artifacts, deformed, ugly, clothes, fabric, garment, texture, watermark, text, bad anatomy, extra limbs, face, hands, fingers"

        # Generate OpenPose control image
        openpose_result = self.openpose(src_image)
        
        # dict 형태일 경우 image 키 추출
        if isinstance(openpose_result, dict):
            openpose_image = openpose_result.get("image")
        else:
            openpose_image = openpose_result
        
        # 검증: 이미지가 정상적으로 반환되었는지 확인
        if not isinstance(openpose_image, Image.Image):
            raise TypeError(f"OpenPose에서 반환된 control image가 유효하지 않습니다: {type(openpose_image)}")
        
        # 이후 파이프라인 호출

        generator = torch.Generator(device="cuda").manual_seed(seed)

        # Use the dedicated skin inpainting pipeline
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
            guidance_scale=5.0  # Lower guidance to better match image context
        ).images[0]

        # Explicitly composite the generated skin onto the original image
        # to ensure only the masked area is affected.
        src_np = np.array(src_image)
        mask_np = np.array(inpaint_mask_img.convert("L")) / 255.0
        mask_np = np.expand_dims(mask_np, axis=-1)
        generated_np = np.array(generated_image)

        # Combine the original image (outside the mask) with the generated skin (inside the mask)
        final_np = src_np * (1 - mask_np) + generated_np * mask_np
        final_image = Image.fromarray(final_np.astype(np.uint8))

        return final_image

    def skin_test(
        self,
        src_image_path: str,
        ref_image_path: str,  # 사용되지 않지만 호환성을 위해 유지
        control_type: str,    # 사용되지 않지만 호환성을 위해 유지
        output_path: str = None,
        step: int = 40,
        seed: int = 42,
        cross_attention_kwargs={"scale": 3},
        vt_model_type: str = "viton_hd",      # 사용되지 않지만 호환성을 위해 유지
        vt_garment_type: str = "upper_body",# 사용되지 않지만 호환성을 위해 유지
        vt_repaint: bool = False,             # 사용되지 않지만 호환성을 위해 유지
        ref_acceleration: bool = False,       # 사용되지 않지만 호환성을 위해 유지
        src_mask_path: str = None             # 사용되지 않지만 호환성을 위해 유지
    ):
        """
        이미지에서 팔과 다리의 옷을 제거하고 사실적인 피부로 인페인팅합니다.
        Removes clothing from arms and legs in an image and inpaints with realistic skin.
        """
        # 1. 이미지 로드 및 준비
        src_image = Image.open(src_image_path).convert("RGB")
        src_image = resize_and_center(src_image, 768, 1024)

        # 2. 원본 이미지에서 의상 마스크 추출
        garment_mask_img = self.mask_predictor(src_image, "overall")["mask"]
        garment_mask_np = np.array(garment_mask_img.convert("L")) > 128

        # 3. 휴먼 파싱을 통해 팔과 다리 마스크 추출
        parsing_map, _ = self.parsing(src_image.resize((768, 1024)))
        parsing_map = np.array(parsing_map)
        limb_mask_raw = np.isin(parsing_map, [4, 5]).astype(np.uint8) # 팔(4), 다리(5)
        limb_mask_img = Image.fromarray(limb_mask_raw * 255).resize(src_image.size, Image.NEAREST)
        limb_mask_np = np.array(limb_mask_img.convert("L")) > 128

        # 4. 인페인팅할 마스크 계산 (의상과 팔/다리가 겹치는 영역)
        inpaint_mask_np = garment_mask_np & limb_mask_np
        inpaint_mask_img = Image.fromarray(inpaint_mask_np.astype(np.uint8) * 255)

        # 인페인팅할 영역이 없으면 원본 이미지와 빈 마스크 반환
        if not np.any(inpaint_mask_np):
            if output_path:
                src_image.save(output_path)
            return src_image, Image.fromarray(np.zeros_like(inpaint_mask_np, dtype=np.uint8) * 255)

        # 5. 피부 생성 호출
        final_image = self.generate_skin(
            src_image=src_image,
            inpaint_mask_img=inpaint_mask_img,
            step=step,
            seed=seed
        )

        if output_path:
            final_image.save(output_path)

        return final_image, inpaint_mask_img

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

        full_body_mask = self.mask_predictor(src_image, "overall")["mask"]

        skin_mask = full_body_mask

        skin_image = self.generate_skin(src_image, full_body_mask, step=step, seed=seed)
        skin_image = skin_image.resize(src_image.size)

        src_np = np.array(src_image)
        mask_np = np.array(full_body_mask.convert("L")) / 255.0
        mask_np = np.expand_dims(mask_np, axis=2)
        skin_np = np.array(skin_image)

        agnostic_np = src_np * (1 - mask_np) + skin_np * mask_np
        agnostic_image = Image.fromarray(agnostic_np.astype(np.uint8))
        
        # 6. 메인 마스크 생성
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
    
        # 7. DensePose 생성
        agnostic_np = np.array(agnostic_image)
        if vt_model_type == "viton_hd":
            seg = self.densepose_predictor.predict_seg(agnostic_np)[:, :, ::-1]
        else:
            iuv = self.densepose_predictor.predict_iuv(agnostic_np)
            seg = np.concatenate([iuv[:, :, :1]] * 3, axis=-1)
        densepose = Image.fromarray(seg)
    
        # 8. 최종 가상 피팅
        transform = LeffaTransform()
        data = {
            "src_image": [agnostic_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
    
        inference = self.vt_inference_hd if vt_model_type == "viton_hd" else self.vt_inference_dc
        
        garment_prompt = "high quality clothing, detailed fabric, realistic texture"
        negative_prompt = "deformed, blurry, low quality, artifacts"
        
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
    
        # 9. 결과 반환
        if output_path:
            gen_image.save(output_path)
        
        return gen_image, mask, densepose, full_body_mask, agnostic_image, skin_image
    
    def leffa_predict_old(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=False,
        output_path: str = None,
        step=30,
        cross_attention_kwargs={"scale": 3},
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        src_mask_path=None  # 선택적 경로 저장
    ):
        assert control_type in ["virtual_tryon", "pose_transfer"], f"Invalid control type: {control_type}"
        
        src_image = Image.open(src_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")

        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        # Mask 생성
        if control_type == "virtual_tryon":
            # vt_garment_type을 AutoMasker의 mask_type에 매핑
            if vt_garment_type == "dresses":
                garment_type_hd = "overall"  # AutoMasker에서 허용되는 값으로 매핑
            elif vt_garment_type == "upper_body":
                garment_type_hd = "upper"
            elif vt_garment_type == "lower_body":
                garment_type_hd = "lower"
            else:
                raise ValueError(f"Invalid vt_garment_type: {vt_garment_type}")

            mask = self.mask_predictor(src_image, garment_type_hd)["mask"]

            if src_mask_path:
                mask.save(src_mask_path)

        elif control_type == "pose_transfer":
            mask = Image.fromarray(np.ones_like(src_image_array, dtype=np.uint8) * 255)

        # DensePose
        if vt_model_type == "viton_hd":
            seg = self.densepose_predictor.predict_seg(src_image_array)[:, :, ::-1]
        else:
            iuv = self.densepose_predictor.predict_iuv(src_image_array)
            seg = np.concatenate([iuv[:, :, :1]] * 3, axis=-1)
        densepose = Image.fromarray(seg)

        # Transform 및 inference
        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)

        inference = self.vt_inference_hd if vt_model_type == "viton_hd" else self.vt_inference_dc
        result = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            cross_attention_kwargs={"scale": cross_attention_kwargs},  # scale을 cross_attention_kwargs로 전달
            seed=seed,
            repaint=vt_repaint
        )

        gen_image = result["generated_image"][0]

        torch.cuda.empty_cache()

        if output_path:
            gen_image.save(output_path)
        
        return gen_image, mask, densepose