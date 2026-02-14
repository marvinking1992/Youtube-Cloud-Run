import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI()
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    detected_language: str
    spanish_translation: str

@app.post("/translate")
async def translate(request: TranslationRequest):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Translate this to Spanish: {request.text}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                # This schema guarantees valid JSON output for the client
                response_schema=TranslationResponse
            )
        )
        return response.parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
