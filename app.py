import os
import shutil
import uuid
import time

from typing import List, Dict, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from modelos.videomae import VideoMAEDetector
from modelos.yolov8_binario import YOLOv8BinarioDetector
from modelos.pose_xgb_booster import PoseXGBBoosterDetector
from modelos.pose_svm import PoseSVMDetector
from modelos.pose_xgb_norm import PoseXGBNormDetector


# ============================================================
# MODELOS PARA RECIBIR INTERVALOS REALES DE ROBO
# ============================================================

class IntervaloRobo(BaseModel):
    inicio_seg: float
    fin_seg: float


class CorrerModeloRequest(BaseModel):
    intervalos_robo: List[IntervaloRobo] = Field(default_factory=list)


# ============================================================
# APP
# ============================================================

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
    "exp5": YOLOv8BinarioDetector(nombre="YOLOv8 Exp5", ruta_pesos="pesos/exp5_best.pt"),
    "exp8": YOLOv8BinarioDetector(nombre="YOLOv8 Exp8 (Data Aug)", ruta_pesos="pesos/exp8_best.pt"),
    "pose_xgb_booster": PoseXGBBoosterDetector(),
    "pose_svm": PoseSVMDetector(),
    "pose_xgb_norm": PoseXGBNormDetector(),
}

# Sesiones activas: { session_id: {"ruta": str, "nombre_original": str} }
SESIONES = {}


@app.get("/")
def inicio():
    return {
        "mensaje": "Benchmark API funcionando",
        "modelos_disponibles": list(MODELOS_DISPONIBLES.keys())
    }


@app.get("/modelos")
def listar_modelos():
    """Lista los modelos disponibles para que el frontend los muestre."""
    return {
        "modelos": [
            {"id": mid, "nombre": m.nombre}
            for mid, m in MODELOS_DISPONIBLES.items()
        ]
    }


# ============================================================
# SESIONES DE BENCHMARK
# ============================================================

@app.post("/sesion/iniciar")
async def iniciar_sesion(file: UploadFile = File(...)):
    """
    Sube el video una sola vez.
    Devuelve un session_id que se usa en las siguientes llamadas de benchmark.
    """
    session_id = str(uuid.uuid4())
    nombre_archivo = f"{session_id}_{file.filename}"
    ruta_video = os.path.join(CARPETA_VIDEOS, nombre_archivo)

    with open(ruta_video, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    SESIONES[session_id] = {
        "ruta": ruta_video,
        "nombre_original": file.filename,
    }

    return {
        "session_id": session_id,
        "nombre_video": file.filename,
        "video_url": f"/sesion/{session_id}/video",
    }


@app.get("/sesion/{session_id}/video")
def obtener_video(session_id: str):
    """Devuelve el video original de la sesión para reproducirlo en el frontend."""
    if session_id not in SESIONES:
        raise HTTPException(404, "Sesión no encontrada")

    ruta = SESIONES[session_id]["ruta"]
    if not os.path.exists(ruta):
        raise HTTPException(404, "Video no existe en disco")

    return FileResponse(ruta, media_type="video/mp4")


@app.post("/sesion/{session_id}/correr/{nombre_modelo}")
def correr_modelo(session_id: str, nombre_modelo: str, request: CorrerModeloRequest):
    """
    Corre un modelo sobre el video de la sesión.
    Además calcula métricas si el detector devuelve predicción frame a frame.
    """
    if session_id not in SESIONES:
        raise HTTPException(404, "Sesión no encontrada")

    if nombre_modelo not in MODELOS_DISPONIBLES:
        raise HTTPException(404, f"Modelo '{nombre_modelo}' no disponible")

    ruta_video = SESIONES[session_id]["ruta"]

    if not os.path.exists(ruta_video):
        raise HTTPException(404, "Video no existe en disco")

    try:
        detector = MODELOS_DISPONIBLES[nombre_modelo]
        resultado = detector.procesar_video(ruta_video)

        prediccion_frames = resultado.get("prediccion_frames")
        fps_video = resultado.get("fps_video")

        if prediccion_frames is not None and fps_video is not None:
            intervalos = [
                (i.inicio_seg, i.fin_seg)
                for i in request.intervalos_robo
            ]

            metricas = calcular_metricas(
                prediccion_frames=prediccion_frames,
                fps=fps_video,
                intervalos=intervalos
            )

            resultado["metricas"] = metricas

            # Se elimina porque puede ser una lista enorme
            resultado.pop("prediccion_frames", None)
        else:
            resultado["metricas"] = None
            resultado["advertencia_metricas"] = (
                "Este modelo no devolvió prediccion_frames o fps_video, "
                "por eso no se pudieron calcular métricas."
            )

        return resultado

    except Exception as e:
        raise HTTPException(500, f"Error procesando: {str(e)}")


@app.delete("/sesion/{session_id}")
def cerrar_sesion(session_id: str):
    """Cierra la sesión y borra el video del disco."""
    if session_id not in SESIONES:
        return {"mensaje": "Sesión ya no existe"}

    ruta = SESIONES[session_id]["ruta"]
    del SESIONES[session_id]

    # En Windows OpenCV a veces no libera el archivo de inmediato
    time.sleep(0.3)

    try:
        if os.path.exists(ruta):
            os.remove(ruta)
    except PermissionError:
        print(f"[WARN] No se pudo borrar {ruta} ahora")

    return {"mensaje": "Sesión cerrada"}


# ============================================================
# LIMPIEZA AUTOMÁTICA DE VIDEOS HUÉRFANOS
# ============================================================

@app.on_event("startup")
def limpiar_videos_huerfanos():
    """Al arrancar el servidor, borra videos viejos que se hayan quedado."""
    if not os.path.exists(CARPETA_VIDEOS):
        return

    for archivo in os.listdir(CARPETA_VIDEOS):
        ruta = os.path.join(CARPETA_VIDEOS, archivo)

        try:
            os.remove(ruta)
        except Exception:
            pass

    print("[Startup] Carpeta de videos limpiada")


# ============================================================
# FUNCIÓN PARA CALCULAR MÉTRICAS
# ============================================================

def normalizar_prediccion(valor: str) -> str:
    """
    Normaliza etiquetas por si algún modelo devuelve:
    'normal', 'Normal', 'NORMAL', 'shoplifting', 'SHOPLIFTING', etc.
    """
    if valor is None:
        return "Normal"

    valor = str(valor).strip().lower()

    if valor in ["shoplifting", "robo", "robbery", "sospechoso", "suspicious"]:
        return "SHOPLIFTING"

    return "Normal"


def calcular_metricas(
    prediccion_frames: List[str],
    fps: float,
    intervalos: List[Tuple[float, float]]
) -> Dict:
    """
    Compara las predicciones frame a frame contra los intervalos reales de robo.

    Retorna:
    - accuracy
    - precision
    - recall
    - f1_score
    - matriz de confusión
    """
    total = len(prediccion_frames)

    if total == 0 or fps <= 0:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "matriz_confusion": {
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0,
            },
            "mensaje": "No hay frames o FPS inválido."
        }

    verdad_frames = []

    for i in range(total):
        tiempo = i / fps

        es_robo = any(
            inicio <= tiempo <= fin
            for inicio, fin in intervalos
        )

        verdad_frames.append("SHOPLIFTING" if es_robo else "Normal")

    prediccion_frames = [
        normalizar_prediccion(p)
        for p in prediccion_frames
    ]

    TP = sum(
        1 for p, v in zip(prediccion_frames, verdad_frames)
        if p == "SHOPLIFTING" and v == "SHOPLIFTING"
    )

    TN = sum(
        1 for p, v in zip(prediccion_frames, verdad_frames)
        if p == "Normal" and v == "Normal"
    )

    FP = sum(
        1 for p, v in zip(prediccion_frames, verdad_frames)
        if p == "SHOPLIFTING" and v == "Normal"
    )

    FN = sum(
        1 for p, v in zip(prediccion_frames, verdad_frames)
        if p == "Normal" and v == "SHOPLIFTING"
    )

    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "matriz_confusion": {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
        },
        "total_frames": total,
        "fps_video": fps,
        "intervalos_evaluados": intervalos,
    }