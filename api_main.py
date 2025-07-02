from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import StreamingResponse
from io import BytesIO
import os
from vton_script import LeffaVirtualTryOn
from fastapi.middleware.cors import CORSMiddleware

print("api 실행")

app = FastAPI()

# API 키
API_KEY = "2004"

# 인증 함수
def api_key_auth(api_key: str = Header(...)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API Key")

vton = LeffaVirtualTryOn(ckpt_dir="./ckpts")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (필요에 따라 제한 가능)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/virtual-tryon")
async def virtual_tryon(src_image: UploadFile = File(...), ref_image: UploadFile = File(...)):
    if not src_image or not ref_image:
        raise HTTPException(status_code=422, detail="Both src_image and ref_image must be provided.")

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)

    src_img_path = os.path.join(temp_dir, src_image.filename)
    ref_img_path = os.path.join(temp_dir, ref_image.filename)

    try:
        with open(src_img_path, "wb") as f:
            f.write(await src_image.read())

        # 기본 참조 이미지 처리
        if ref_image.filename == "default_ref_image.jpg":
            ref_img_path = os.path.join("default_images", "default_ref_image.jpg")
        else:
            with open(ref_img_path, "wb") as f:
                f.write(await ref_image.read())

        output_image, mask, densepose = vton.leffa_predict(
            src_image_path=src_img_path,
            ref_image_path=ref_img_path,
            control_type="virtual_tryon",
            vt_model_type="viton_hd",
            vt_garment_type="upper_body",
            vt_repaint=True,
            output_path="output/virtual_tryon_result.jpg"
        )

        img_io = BytesIO()
        output_image.save(img_io, format="JPEG")
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
    finally:
        if os.path.exists(src_img_path):
            os.remove(src_img_path)
        if os.path.exists(ref_img_path) and ref_image.filename != "default_ref_image.jpg":
            os.remove(ref_img_path)