from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
import google.generativeai as genai
from io import BytesIO
from PIL import Image
import os

load_dotenv()

app = FastAPI()

def get_content(image: Image):
    api_key = os.getenv("OCR_API_KEY")
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction="Your task is to only read the image provided and generate the content written on it. \
Do not perform any other task apart from this. If no text is found on the image, do not generate anything."
    )

    text = model.generate_content(["", image])

    return text.text


@app.post("/ocr/")
async def extract_text_from_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        extracted_text = get_content(image)

        return { "content" : extracted_text}

    except Exception as e:
        return { "error": str(e) }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)