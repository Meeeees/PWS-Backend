import json
import os
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
shared_frame = io.BytesIO()

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    global picam2
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")

    picam2.configure(camera_config)
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
    global shared_frame
    if not picam2:
        raise HTTPException(status_code=500, detail="Camera not initialized.")
    async def generate_frames():
        while True:
            await asyncio.sleep(0.01)
            with camera_lock:
                temp_frame = io.BytesIO()  # Temporary frame container
                picam2.capture_file(temp_frame, format="jpeg")
                temp_frame.seek(0)
                shared_frame.seek(0)
                shared_frame.truncate(0)  # Clear shared frame
                shared_frame.write(temp_frame.read())  # Update shared frame
                shared_frame.seek(0)  # Reset pointer for reading
            
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + shared_frame.read() + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

frame = 0

@app.get("/frame_counter")
async def frame_counter():
    global frame
    return frame

@app.get("/start")
async def start(name: str | None = None):
    global StreamMain
    global frame
    if name == "":
        return {
            "status": 400,
            "reden": "Geef alsjeblieft een naam op"
        }
    if os.path.exists(f"frame/{name}"):
        return {
            "status": 409,
            "reden": "Die naam is al door iemands anders in gebruik"
        }
    os.mkdir(f"frame/{name}")
    frame = 1;
    print("Start")
    while frame < 51:   
        with camera_lock:
            current_frame = io.BytesIO(shared_frame.getvalue()) 
        print(frame)

        with open (f"frame/{name}/frame_{frame}.png", "wb") as f: 
            f.write(current_frame.getvalue())
        frame = frame + 1
        await asyncio.sleep(0.1)
        

@app.get("/recognize")
async def recognize(request: Request):
    print("Connection made to recognize")
    streams = {
        1: io.BytesIO(),
        2: io.BytesIO(),
        3: io.BytesIO(),
        4: io.BytesIO(),
        5: io.BytesIO(),
    }
    with camera_lock:
        for i in range(1, 6):
            picam2.capture_file(streams.get(i), format="jpeg")
            streams.get(i).seek(0)

    tried = 1
    result = 1
    while result == 1 and tried <= 5:
        current_stream = streams.get(tried)
        if current_stream:
            result = await recognize_image(current_stream)
        tried += 1

    return result
        


async def recognize_image(stream):
    image = Image.open(stream)
    imgArray = np.array(image)
    try: 
        results = DeepFace.find(img_path=imgArray, db_path="frame",  enforce_detection=True)  
    except:
        print("Opnieuw proberen")
        return 1

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
