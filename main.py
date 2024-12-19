import os
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
import json
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from deepface import DeepFace
import shutil
import re
import io
import sys
from picamera2 import Picamera2
import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from starlette.responses import StreamingResponse
from datetime import datetime
import pandas as pd
import time

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=5)
picam2 = None
camera_lock = Lock()
StartTijd = 0
shared_frame = io.BytesIO()
NeedToTrain = False;
Progress = 0
frame = 0

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
    picam2.set_controls({"NoiseReductionMode": 4})

    picam2.start()

    asyncio.create_task(capture_frames())

@app.on_event("shutdown")
async def shutdown_event():
    with open("TijdAnalyse.txt", "a") as file:
        file.write(f"\n")

    global picam2
    if picam2:
        picam2.stop()

@app.get("/")
async def root():
    return {"message": "Please make a request to '/predict'"}

security = HTTPBasic()

def Beveiliging(request: Request):
    username = request.query_params.get("username")
    password = request.query_params.get("password")
    
    if username != "admin" or password != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

def capture_frame():
    global shared_frame
    with camera_lock:
        temp_frame = io.BytesIO()
        picam2.capture_file(temp_frame, format="jpeg")
        temp_frame.seek(0)
        shared_frame.seek(0)
        shared_frame.truncate(0)
        shared_frame.write(temp_frame.read())
        shared_frame.seek(0)

async def capture_frames():
    while True:
        capture_frame()
        await asyncio.sleep(0.05)

@app.get("/video", dependencies=[Depends(Beveiliging)])
async def video_stream():
    async def generate_frames():
        while True:
            await asyncio.sleep(0.05)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + shared_frame.getvalue() + b'\r\n')

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/gezichten/{path:path}")
async def gezichten(path: str):
    file_path = os.path.join("gezichten", path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/herkend/{path:path}")
async def herkend(path: str):
    file_path = os.path.join("herkend", path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

@app.get("/frame_counter")
async def frame_counter():
    global frame
    return frame

@app.get("/info", dependencies=[Depends(Beveiliging)])
async def info(type: int | None = None):
    if type == 0:
        return {"NeedToTrain": NeedToTrain}
    elif type == 1:
        return {Progress}
    elif type == 2:
        mensen = os.listdir("gezichten")
        gefilterd = [mens for mens in mensen if not mens.endswith('0.pkl')]
        mensen = {mens: [] for mens in gefilterd}
        for mens in mensen:
            mensen[mens].append(os.listdir(f"gezichten/{mens}"))

        return mensen
    elif type == 3:
        if StartTijd == 0:
            return "No image recognition yet"
        return {"verschil": time.time() - StartTijd}
    
@app.get("/start", dependencies=[Depends(Beveiliging)])
async def start(name: str | None = None):
    global NeedToTrain
    global StreamMain
    global frame
    if name == "":
        return {
            "status": 400,
            "reden": "Geef alsjeblieft een naam op"
        }

    if os.path.exists(f"gezichten/{name}"):
        frames = os.listdir(f"gezichten/{name}")
        numbers = [int(re.search(r'(\d+)(?=\D*$)', frame).group()) for frame in frames]
        numbers.sort(reverse=True)
        frame = numbers[0] + 1
    else:
        os.mkdir(f"gezichten/{name}")
        frame = 1;
    startFrame = frame

    print("Start")
    while frame < startFrame + 50:   
        with camera_lock:
            current_frame = io.BytesIO(shared_frame.getvalue()) 
        print(frame)

        with open (f"gezichten/{name}/frame_{frame}.png", "wb") as f: 
            f.write(current_frame.getvalue())
        frame = frame + 1
        await asyncio.sleep(0.1)
    NeedToTrain = True
    return "Success"
        
@app.get("/verwijder", dependencies=[Depends(Beveiliging)])
async def verwijder(naam: str, frame: str | None = None):
    print(naam, frame)
    if not re.match(r"^[a-zA-Z0-9_-]+$", naam):
        raise HTTPException(status_code=400, detail="Incorrecte naam")
    if frame:
        if not re.match(r"^[a-zA-Z0-9._-]+$", frame):
            raise HTTPException(status_code=400, detail="Incorrecte frame")
        os.remove(f"gezichten/{naam}/{frame}")
        return "Verwijderd"
    shutil.rmtree(f"gezichten/{naam}")
    return "Verwijderd"


@app.get("/recognize", dependencies=[Depends(Beveiliging)])
async def recognize(request: Request):
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
    if result == 1:
        return {"status": 422, "reden": "Er kon geen gezicht worden herkend"}
    return {"status": 200, "result": result}
        

def OutputHandle(message):
    global Progress
    if "Face could not be detected" in message:
        file = message.split("Exception while extracting faces from")[1].split(": Face could")[0].strip()
        if os.path.exists(file):
            os.remove(file)
            sys.__stdout__.write(f"verwijderd: {file}\n")
    if "Finding representations" in message:
        match = re.search(r"(\d+)%", message)
        if match:
            procent = match.group(1)
            sys.__stdout__.write(f"Progress: {procent}%\n")
            Progress = procent

class SystemOutputHandle:
    def write(self, message):
        OutputHandle(message)
        
    def flush(self):
        sys.__stdout__.flush()

async def recognize_image(stream, byPass = False, byPassPath = ""):
    global StartTijd

    def perform_deepface_find(img_array):
        global NeedToTrain
        global EindTijd
        global StartTijd
        try:
            print("Gezichts herkenning gaat plaatsvinden")
            sys.stdout = SystemOutputHandle()
            sys.stderr = sys.stdout

            if not byPass:
                results = DeepFace.find(img_path=img_array, db_path="gezichten")
            if byPass:
                results = DeepFace.find(img_path=byPassPath, db_path="gezichten")
            

            NeedToTrain = False

            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

            return results
        except Exception as e:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"DeepFace error: {e}")
            return None

    image = Image.open(stream)
    img_array = np.array(image)

    StartTijd = time.time()
    results = await asyncio.get_event_loop().run_in_executor(executor, perform_deepface_find, img_array)
    EindTijd = time.time()

    if results is None:
        print("Failed to recognize")
        return 1  


    if results and isinstance(results, list):
        df = pd.concat(results, ignore_index=True)
    else:
        df = pd.DataFrame(results) 
    print("Result: ", results)
    resultaten = df.to_dict()
    print(resultaten)

    ResultsFrameNamenDict = resultaten["identity"]
    ResultatenLijst = list(ResultsFrameNamenDict.values())



    Personen = {}
    print("Naar for loop")
    for resultaat in ResultatenLijst:
        naam = resultaat.split("/")[1]
        if naam not in Personen:
            Personen[naam] = [resultaat]
        else:
            Personen[naam].append(resultaat)
    try:
        font = ImageFont.truetype("SUSE-Regular.ttf", 40)
    except IOError:
        font = None

    print("Herkende persoon(en): ", Personen)

    percentage = 0
    success = False
    percentages = []
    for persoon, frames in Personen.items():
        Hoeveelheid = len(frames)
        MaxHoeveelheid = len(os.listdir(f"gezichten/{persoon}"))
        percentages.append({persoon: Hoeveelheid/MaxHoeveelheid})
        if Hoeveelheid > MaxHoeveelheid:
            Hoeveelheid = MaxHoeveelheid
        print(Hoeveelheid, MaxHoeveelheid, Hoeveelheid/MaxHoeveelheid)
        if Hoeveelheid/MaxHoeveelheid > percentage:
            percentage = Hoeveelheid/MaxHoeveelheid
        if Hoeveelheid/MaxHoeveelheid > 0.5:
            success = True
            for frame in resultaten["identity"].values():
                if frame.split("/")[1] == persoon:
                    index = next((key for key, value in resultaten["identity"].items() if value == frame), None)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((resultaten["source_x"][index], resultaten["source_y"][index],
                        resultaten["source_x"][index] + resultaten["source_w"][index],
                        resultaten["source_y"][index] + resultaten["source_h"][index]),
                       outline="red", width=2)
                    text_position = (resultaten["source_x"][index], resultaten["source_y"][index])
                    if font:
                        draw.text(text_position, f"{persoon}, {round(Hoeveelheid/MaxHoeveelheid, 3) * 100}%", fill="red", font=font)
                    else:
                        draw.text(text_position, persoon, fill="red")
                    break
            
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    FileName = f"herkend/herkend_op_{datetime.now().strftime('%d-%m-%Y-%H:%M:%S')}.jpg"
    with open(FileName, "wb") as f:
        f.write(buffered.getvalue())

    TijdVerschil = EindTijd - StartTijd
    tijd_analyse_data = {
        "Tijd nodig": TijdVerschil,
        "Succesvol herkend": success,
        "percentages": percentages
    }

    tijd_analyse_file = "accuratieAnalyse.json"
    with open(tijd_analyse_file, "r") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            data = []

    data.append(tijd_analyse_data)

    with open(tijd_analyse_file, "w") as file:
        json.dump(data, file, indent=4)

    return [FileName, percentage]

@app.get("/accuratieTestStart")
async def AccuratieTest():
    dummyStream = io.BytesIO()
    print("AccuratieTestStart")
    with camera_lock:
        picam2.capture_file(dummyStream, format="jpeg")
        dummyStream.seek(0)
    files = os.listdir("fotoAccuratieTest")
    print(files)
    results = []
    for file in files:
        print(f"Nu {file} testen")
        result = await recognize_image(dummyStream, True, "fotoAccuratieTest/" + file)
        results.append(result)