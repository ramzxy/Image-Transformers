import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
from Transformer import ImageTransformer

app = FastAPI()

class ImageRequest(BaseModel):
    image_path: str

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/transform")
def transform_image(request: ImageRequest):
    transformer = ImageTransformer(request.image_path)
    transformer.transform()
    return {"message": "Image transformed successfully!"}