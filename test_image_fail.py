from config import CONFIG
from engine import process_file_input
from models import AIProcessingError
import traceback

with open("dummy.png", "wb") as f:
    f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDAT\x08\x99c\xf8\x0f\x04\x00\x09\xfb\x03\xfd\xe3U\xf2\x9c\x00\x00\x00\x00IEND\xaeB`\x82')

try:
    res = process_file_input("dummy.png", "image/png", "test context")
    print("SUCCESS")
except Exception as e:
    print("ERROR CAUGHT")
    traceback.print_exc()
