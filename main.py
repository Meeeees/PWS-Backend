import json
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
# from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace
import pandas as pd
import logging
import base64
import io
from fastapi.responses import StreamingResponse
from picamera2 import Picamera2, Preview
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from starlette.responses import StreamingResponse
from datetime import datetime


app = FastAPI()
executor = ThreadPoolExecutor(max_workers=5)
picam2 = None  # Global camera instance
camera_lock = Lock()  # Lock to ensure thread safety

# CORS setup
origins = [
    "http://localhost:8000", 
    "http://127.0.0.1:8000",
    "http://localhost:5000",
    "http://localhost:3000",

]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global picam2
    picam2 = Picamera2()
    picam2.start()  # Start the camera

@app.on_event("shutdown")
async def shutdown_event():
    global picam2
    if picam2:
        picam2.stop()  # Stop the camera



# @app.get("/img")
# async def img():
#     if not picam2:
#         raise HTTPException(status_code=500, detail="Camera not initialized.")

#     with camera_lock: 
#         try:
#             loop = asyncio.get_running_loop()
#             Buffer = await loop.run_in_executor(executor, capture_image)
#             recognize = await recognize_image(Buffer)
#             return StreamingResponse(recognize, media_type="image/jpeg")
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict(request: Request):
    print("Connection made to predict")
    body = await request.body()
    Buffer = body.decode("utf-8")
    cleanBuffer = Buffer.replace("data:image/jpeg;base64,", "")
    newBuffer = await recognize_image(cleanBuffer)

    return {"newBuffer": newBuffer}

    
# def capture_image():
#     stream = io.BytesIO()
#     picam2.capture_file(stream, format="jpeg")
#     stream.seek(0)
#     image = Image.open(stream)
#     return image

async def generate_frames():
    while True:
        await asyncio.sleep(0.01)
        with camera_lock:
            stream = io.BytesIO()
            picam2.capture_file(stream, format="jpeg")
            stream.seek(0)

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + stream.read() + b'\r\n')

@app.get("/video")
async def video_stream():
    if not picam2:
        raise HTTPException(status_code=500, detail="Camera not initialized.")

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
async def root():
    return {"message": "Please make a request to '/predict'"}


@app.get("/recognize")
async def recognize(request: Request):
    print("Connection made to recognize")
    with camera_lock:
        stream = io.BytesIO()
        picam2.capture_file(stream, format="jpeg")
        stream.seek(0)
        print("going to cal lrecognize")
        await recognize_image(stream)
    
    return {"status": "Frame captured and saved"}
        


async def recognize_image(stream):
    print("gonna recognize")
    image = Image.open(stream)
    imgArray = np.array(image)
    results = DeepFace.find(img_path=imgArray, db_path="images",  enforce_detection=False)  
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

    print(results)
    for person in results:
        if len(person["target_x"]) == 0:
            continue
        draw.rectangle((person["source_x"][0], person["source_y"][0], person["source_x"][0] + person["source_w"][0], person["source_y"][0] + person["source_h"][0]), outline="red", width=2)
        text = person["identity"][0].split("/")[1]
        text_position = (person["source_x"][0], person["source_y"][0])
        if font:
            draw.text(text_position, text, fill="red", font=font)
        else:
            draw.text(text_position, text, fill="red")

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    with open("output_image.jpg", "wb") as f:  # 'wb' means write in binary mode
        f.write(buffered.getvalue())
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")


    return base64_image  
