import sys
import time
from io import StringIO

class LiveLogger:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.buffer = StringIO()

    def write(self, message):
        self.original_stream.write(f"JO: {message}")  # Print to terminal
        self.original_stream.flush()         # Ensure it appears live
        self.buffer.write(message)          # Save to buffer

    def flush(self):
        self.original_stream.flush()

    def getvalue(self):
        return self.buffer.getvalue()

# Simulate a long-running function
def long_running_function():
    print("Step 1: Initializing...")
    time.sleep(2)
    print("Step 2: Processing data...")
    time.sleep(2)
    print("Step 3: Finalizing...")
    time.sleep(2)
    print("Done!")

# Redirect stdout to a LiveLogger
original_stdout = sys.stdout
logger = LiveLogger(original_stdout)
sys.stdout = logger

long_running_function()  # Output will print live and be logged

sys.stdout = original_stdout  # Restore original stdout

# Access captured logs
logged_output = logger.getvalue()
print("\nCaptured Logs:")
print(logged_output)
