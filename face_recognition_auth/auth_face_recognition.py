from fastapi import FastAPI, HTTPException,APIRouter
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import face_recognition

router = APIRouter()
class FaceImage(BaseModel):
    image: str  # base64 string

def decode_base64_image(base64_str: str):
    try:
        base64_data = base64_str.split(",")[-1]
        image_data = base64.b64decode(base64_data)
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format")

@router.post("/generate-descriptor")
def generate_descriptor(data: FaceImage):
    img = decode_base64_image(data.image)
    face_encodings = face_recognition.face_encodings(img)

    if not face_encodings:
        raise HTTPException(status_code=400, detail="No face detected")

    return {"descriptor": face_encodings[0].tolist()}
