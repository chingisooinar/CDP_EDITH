from typing import Optional
from fastapi import FastAPI
import Models
from io import BytesIO
from starlette.responses import StreamingResponse

app = FastAPI()


@app.get("/generate")
def GenerateSketch():
    request = 'anime.jpg'
    response = Models.generateSketch(request)

    return StreamingResponse(BytesIO(response.tobytes()), media_type="image/png")


@app.get("/colorize")
def Colorize():
    response = Models.colorizeModel()
    return response