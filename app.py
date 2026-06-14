import os
import shutil
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from modelos.videomae import VideoMAEDetector


app = FastAPI(title="Benchmark de Modelos de Detección de Robos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CARPETA_VIDEOS = "videos_benchmark"
os.makedirs(CARPETA_VIDEOS, exist_ok=True)

MODELOS_DISPONIBLES = {
    "videomae": VideoMAEDetector(),
}


@app.get("/")
def inicio():
    return {
        "mensaje": "Benchmark API funcionando",
        "modelos_disponibles": list(MODELOS_DISPONIBLES.keys())
    }


@app.post("/benchmark/{nombre_modelo}")
async def correr_benchmark(nombre_modelo: str, file: UploadFile = File(...)):
    if nombre_modelo not in MODELOS_DISPONIBLES:
        raise HTTPException(404, f"Modelo '{nombre_modelo}' no disponible")

    nombre_unico = f"{uuid.uuid4()}_{file.filename}"
    ruta_video = os.path.join(CARPETA_VIDEOS, nombre_unico)
    with open(ruta_video, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        detector = MODELOS_DISPONIBLES[nombre_modelo]
        resultado = detector.procesar_video(ruta_video)
        return resultado
    except Exception as e:
        raise HTTPException(500, f"Error procesando: {str(e)}")
    finally:
        if os.path.exists(ruta_video):
            os.remove(ruta_video)