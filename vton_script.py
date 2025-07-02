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
from diffusers import StableDiffusionInpaintPipeline

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

        # majicmix realistic skin model - diffusers pipeline (xformers 비활성화)
        # self.majic_pipe = StableDiffusionInpaintPipeline.from_single_file(
        #     f"{ckpt_dir}/majicmixRealistic_v7.safetensors",
        #     torch_dtype=torch.float16,
        #     safety_checker=None
        # ).to("cuda")

    def generate_skin(self, image: Image.Image, mask: Image.Image, vt_model_type: str):
        try:
            # 1. Human Parsing 수행
            parsing_map, _ = self.parsing(image.resize((384, 512)))
            parsing_map = np.array(parsing_map)

            # 2. 얼굴 부분 추출하여 평균 피부색 계산 (ATR 기준 얼굴=1)
            face_mask = (parsing_map == 1).astype(np.uint8)
            face_mask_resized = Image.fromarray(face_mask * 255).resize(image.size, Image.BILINEAR)
            face_mask_np = np.array(face_mask_resized) / 255.0
            face_np = np.array(image) * face_mask_np[:, :, None]

            valid_pixels = face_mask_np[:, :, None].sum()
            if valid_pixels > 0:
                skin_color = np.sum(face_np, axis=(0, 1)) / (valid_pixels + 1e-5)
            else:
                skin_color = np.array([180, 140, 120], dtype=np.uint8)  # fallback tone

            # 3. 팔/다리 마스크 추출 (팔=4, 다리=5)
            limb_mask = np.isin(parsing_map, [4, 5]).astype(np.uint8)
            limb_mask_img = Image.fromarray(limb_mask * 255).resize(image.size, Image.BILINEAR)

            # 4. 색 채우기
            skin_color = np.clip(skin_color, 0, 255).astype(np.uint8)
            base = np.array(image)

            # 밝기 노이즈 추가 (자연스러움 향상)
            noise = np.random.normal(loc=1.0, scale=0.03, size=base.shape)
            skin_array = (np.ones_like(base, dtype=np.float32) * skin_color[None, None, :]) * noise
            skin_array = np.clip(skin_array, 0, 255).astype(np.uint8)

            limb_mask_np = np.expand_dims(np.array(limb_mask_img) / 255.0, axis=2)
            result_np = base * (1 - limb_mask_np) + skin_array * limb_mask_np
            result_image = Image.fromarray(result_np.astype(np.uint8))
            return result_image

        except Exception as e:
            print(f"❌ Skin generation failed: {e}")
            return Image.new("RGB", image.size, (255, 255, 255)), None

    def skin_test(
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

        skin_image = self.generate_skin(src_image, full_body_mask, vt_model_type)

        src_np = np.array(src_image)
        mask_np = np.array(full_body_mask.convert("L")) / 255.0
        mask_np = np.expand_dims(mask_np, axis=2)
        skin_np = np.array(skin_image)

        agnostic_np = src_np * (1 - mask_np) + skin_np * mask_np
        agnostic_image = Image.fromarray(agnostic_np.astype(np.uint8))

        return agnostic_image, skin_image

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
        
        # 1. 이미지 로드 및 리사이징
        src_image = Image.open(src_image_path).convert("RGB")
        ref_image = Image.open(ref_image_path).convert("RGB")
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)
        
        # 2. 전신 마스크 생성 (옷 제거용)
        full_body_mask = self.mask_predictor(src_image, "overall")["mask"]
        
        # 3. 피부 생성용 마스크 준비 (반전 제거)
        skin_mask = full_body_mask
        
        # 4. 디퓨전으로 피부 생성 (vt_model_type 전달)
        skin_image = self.generate_skin(src_image, skin_mask, vt_model_type)
        
        # 5. 옷 제거 및 피부 적용
        src_np = np.array(src_image)
        mask_np = np.array(full_body_mask.convert("L")) / 255.0
        mask_np = np.expand_dims(mask_np, axis=2)
        skin_np = np.array(skin_image)
        
        # 수정된 블렌딩: 피사체 영역에 생성된 피부 적용
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
            cross_attention_kwargs={"scale": cross_attention_kwargs},
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



        #=========
        # vton_script.py

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
