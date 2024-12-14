import time
import sys
import os
def function():
    print("24-12-14 18:57:04 - 🔴 Exception while extracting faces from frame1.png: Face could not be detected in gezichten/henkie/frame_27.png.Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")
    time.sleep(2)
    print("24-12-14 18:57:04 - 🔴 Exception while extracting faces from frame2.png: Face could not be detected in gezichten/henkie/frame_27.png.Please confirm that the picture is a face photo or consider to set enforce_detection param to False.")

def handle(message):
    sys.__stdout__.write("HI\n")
    if "Face could not be detected" in message:
        file = message.split("Exception while extracting faces from")[1].split(": Face could")[0].strip()
        sys.__stdout__.write(f"{file}, {os.path.exists(file)}\n")
        if os.path.exists(file):
            os.remove(file)


class CustomStream:
    def write(self, message):
        handle(message)
        
    def flush(self):
        sys.__stdout__.flush()

# Override sys.stdout to our custom stream
sys.stdout = CustomStream()

function()

sys.stdout = sys.__stdout__
