import json
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace
import pandas as pd
import logging
import base64
import io
from picamera2 import Picamera2, Preview
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from starlette.responses import StreamingResponse
from datetime import datetime
import calendar

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=5)
picam2 = None
camera_lock = Lock()

app.mount("/herkend", StaticFiles(directory="herkend"), name="herkend")

origins = [
    "http://localhost:8000", 
    "http://127.0.0.1:8000",
    "http://localhost:5000",
    "http://localhost:3000",
    "*"

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
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()


@app.on_event("shutdown")
async def shutdown_event():
    global picam2
    if picam2:
        picam2.stop()


@app.get("/")
async def root():
    return {"message": "Please make a request to '/predict'"}

@app.get("/video")
async def video_stream():
    if not picam2:
        raise HTTPException(status_code=500, detail="Camera not initialized.")
    async def generate_frames():
        while True:
            await asyncio.sleep(0.01)
            with camera_lock:
                stream = io.BytesIO()
                picam2.capture_file(stream, format="jpeg")
                stream.seek(0)

                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + stream.read() + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")




@app.get("/start")
async def start():
    StartTime = calendar.timegm(datetime.now().timetuple())
    frame = 1;
    while StartTime + 5 > calendar.timegm(datetime.now().timetuple()):    
        with camera_lock:
            if frame == 1:
                return "Gestart"
            stream = io.BytesIO()
            picam2.capture_file(stream, format="jpeg")
            stream.seek(0)
            with open (f"frame/frame_{frame}.jpeg", "wb") as f: 
                f.write(stream.getvalue())
            frame = frame + 1
        await asyncio.sleep(0.1)
        

@app.get("/recognize")
async def recognize(request: Request):
    print("Connection made to recognize")
    stream = io.BytesIO()
    with camera_lock:
        picam2.capture_file(stream, format="jpeg")
        stream.seek(0)
        print("going to call recognize")
    
    Result = await recognize_image(stream)
    return Result
        


async def recognize_image(stream):
    print("gonna recognize")
    image = Image.open(stream)
    imgArray = np.array(image)
    print("Now going to run DeepFace")
    results = DeepFace.find(img_path=imgArray, db_path="images",  enforce_detection=False)  
    print("Result: ",results)
    if isinstance(results, list) and len(results) > 0:
        json_results = json.dumps([result.to_dict() if isinstance(result, pd.DataFrame) else result for result in results])
    else:
        json_results = json.dumps(results)

    
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("SUSE-Regular.ttf", 40)
    except IOError:
        font = None

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
    FileName = f"herkend/herkend_op_{datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}.jpg"
    with open(FileName, "wb") as f: 
        f.write(buffered.getvalue())

    return FileName
