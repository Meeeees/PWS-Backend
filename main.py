import json
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace
import pandas as pd
import logging
import base64
import io
from fastapi.responses import StreamingResponse

app = FastAPI()
model = YOLO('yolov8n.pt')
origins = [
    "http://localhost:8000", 
    "http://127.0.0.1:8000",
    "http://localhost:5000",
]
print("hello")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test/")
async def test():
    print("Connection made to test: ")
    return {"message": "Hello World"}


@app.post("/predict/")
async def predict(request: Request):
    print("Connection made to predict")
    body = await request.body()
    Buffer = body.decode("utf-8")
    cleanBuffer = Buffer.replace("data:image/jpeg;base64,", "")
    newBuffer = await recognize_image(cleanBuffer)

    return {"newBuffer": newBuffer}


async def recognize_image(buffer):
    image = Image.open(io.BytesIO(base64.b64decode(buffer)))

    imgArray = np.array(image)
    results = DeepFace.find(img_path=imgArray, db_path="images",  enforce_detection=False, silent=True)  
    print("Result: ",results, "end")
    if isinstance(results, list) and len(results) > 0:
        json_results = json.dumps([result.to_dict() if isinstance(result, pd.DataFrame) else result for result in results])
    else:
        json_results = json.dumps(results)

    print("JSON: ", json_results)
    
    with open('result.json', 'w') as f:
        f.write(json_results)
    print("DID JSON")

    with open('result.txt', 'w') as f:
        for item in results:
             f.write("%s\n" % item)

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("SUSE-Regular.ttf", 40)
    except IOError:
        font = None

    for person in results:
        if len(person["target_x"]) == 0:
            continue
        draw.rectangle((person["source_x"][0], person["source_y"][0], person["source_x"][0] + person["source_w"][0], person["source_y"][0] + person["source_h"][0]), outline="red", width=2)
        text = person["identity"][0].split("\\")[1]
        text_position = (person["source_x"][0], person["source_y"][0])
        if font:
            draw.text(text_position, text, fill="red", font=font)
        else:
            draw.text(text_position, text, fill="red")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")

    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    


    return base64_image  

# Dit is de oude code voor de object herkenning, wordt niet meer gebruikt.
@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    print("Connection made: ")  
    print(file)
    contents = await file.read()
    image = Image.open(io.BytesIO(contents));  
    results = model(image)

    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("SUSE-Regular.ttf", 40)
    except IOError:
        font = None

    for result in results:
        for box in result.boxes:
            xywh = box.xywh.tolist()[0] 

            if len(xywh) >= 4:  
                print("Box:", box.xywh.tolist(), box.cls.item(), box.conf.item())
                x, y, w, h = xywh[0], xywh[1], xywh[2], xywh[3]
                draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], outline="red", width=2)
                class_name = model.names[box.cls.item()]
                confidence = box.conf.item()
                text = f"{class_name} {confidence:.2f}"
                text_position = (x - w / 2, y - h / 2)
                print(text_position)

    
                if font:
                    draw.text(text_position, text, fill="red", font=font)
                else:
                    draw.text(text_position, text, fill="red")
            else:
                print(f"Unexpected format for xywh: {xywh}")

    image.save("server_output.jpg")

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")

    img_size = len(img_byte_arr.getvalue())
    if img_size == 0:
        return {"error": "Empty image byte stream"}

    img_byte_arr.seek(0) 

    return StreamingResponse(img_byte_arr, media_type="image/jpeg")
