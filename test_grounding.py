from config import CONFIG
from google import genai
import sys

client = genai.Client(api_key=CONFIG.GEMINI_API_KEY)
try:
    with open("test.txt", "w") as f:
        f.write("test")
    file = client.files.upload(path="test.txt")
    response = client.models.generate_content(
        model=CONFIG.GEMINI_MODEL,
        contents=["test", file],
        config={"tools": [{"google_search": {}}]}
    )
    print("SUCCESS")
except Exception as e:
    print("ERROR:", e)
