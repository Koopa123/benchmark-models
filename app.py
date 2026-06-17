import os
import json
import shutil
import uuid
import time

from typing import List, Dict, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from modelos.videomae import VideoMAEDetector
from modelos.yolov8_binario import YOLOv8BinarioDetector
from modelos.pose_xgb_booster import PoseXGBBoosterDetector
from modelos.pose_svm import PoseSVMDetector
from modelos.pose_xgb_norm import PoseXGBNormDetector


# ============================================================
# MODELOS PYDANTIC
# ============================================================

class IntervaloRobo(BaseModel):
    inicio_seg: float
    fin_seg: float


class CorrerModeloRequest(BaseModel):
    intervalos_robo: List[IntervaloRobo] = Field(default_factory=list)


class IniciarDemoRequest(BaseModel):
    demo_id: str


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
CARPETA_DEMOS = "videos_demo"
META_DEMOS = os.path.join(CARPETA_DEMOS, "meta.json")

os.makedirs(CARPETA_VIDEOS, exist_ok=True)
os.makedirs(CARPETA_DEMOS, exist_ok=True)

MODELOS_DISPONIBLES = {
    "videomae": VideoMAEDetector(),
    "exp5": YOLOv8BinarioDetector(nombre="YOLOv8 Exp5", ruta_pesos="pesos/exp5_best.pt"),
    "exp8": YOLOv8BinarioDetector(nombre="YOLOv8 Exp8 (Data Aug)", ruta_pesos="pesos/exp8_best.pt"),
    "pose_xgb_booster": PoseXGBBoosterDetector(),
    "pose_svm": PoseSVMDetector(),
    "pose_xgb_norm": PoseXGBNormDetector(),
}

MODELOS_VISUALES = {
    "pose_xgb_booster",
    "pose_xgb_norm",
    "pose_svm",
}

# Sesiones activas: { session_id: {"ruta": str, "nombre_original": str, "es_demo": bool} }
SESIONES = {}


@app.get("/")
def inicio():
    return {
        "mensaje": "Benchmark API funcionando",
        "modelos_disponibles": list(MODELOS_DISPONIBLES.keys())
    }


@app.get("/modelos")
def listar_modelos():
    return {
        "modelos": [
            {"id": mid, "nombre": m.nombre}
            for mid, m in MODELOS_DISPONIBLES.items()
        ]
    }


# ============================================================
# VIDEOS DEMO PRE-CARGADOS
# ============================================================

def _cargar_meta_demos() -> List[Dict]:
    """Carga el meta.json de los videos demo. Si no existe, devuelve []."""
    if not os.path.exists(META_DEMOS):
        return []
    try:
        with open(META_DEMOS, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Error leyendo {META_DEMOS}: {e}")
        return []


@app.get("/videos-demo")
def listar_videos_demo():
    """Lista los videos demo disponibles con sus metadatos."""
    demos = _cargar_meta_demos()
    disponibles = []

    for demo in demos:
        ruta = os.path.join(CARPETA_DEMOS, demo["archivo"])
        if not os.path.exists(ruta):
            # Si el archivo no existe, lo saltamos
            continue
        disponibles.append({
            "id": demo["id"],
            "nombre": demo["nombre"],
            "descripcion": demo.get("descripcion", ""),
            "intervalos_robo": demo.get("intervalos_robo", []),
        })

    return {"videos": disponibles}

@app.get("/sesion-preview-demo")
def preview_demo(demo_id: str):
    """Devuelve el archivo de video de un demo para hacer preview en el frontend."""
    demos = _cargar_meta_demos()
    demo = next((d for d in demos if d["id"] == demo_id), None)
    if not demo:
        raise HTTPException(404, "Video demo no encontrado")

    ruta = os.path.join(CARPETA_DEMOS, demo["archivo"])
    if not os.path.exists(ruta):
        raise HTTPException(404, "Archivo no existe")

    return FileResponse(ruta, media_type="video/mp4")

# ============================================================
# SESIONES DE BENCHMARK
# ============================================================

@app.post("/sesion/iniciar")
async def iniciar_sesion(file: UploadFile = File(...)):
    """Sube un video y crea una sesión nueva."""
    session_id = str(uuid.uuid4())
    nombre_archivo = f"{session_id}_{file.filename}"
    ruta_video = os.path.join(CARPETA_VIDEOS, nombre_archivo)

    with open(ruta_video, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    SESIONES[session_id] = {
        "ruta": ruta_video,
        "nombre_original": file.filename,
        "es_demo": False,
    }

    return {
        "session_id": session_id,
        "nombre_video": file.filename,
        "video_url": f"/sesion/{session_id}/video",
    }


@app.post("/sesion/iniciar-demo")
def iniciar_sesion_demo(request: IniciarDemoRequest):
    """
    Crea una sesión usando un video demo pre-cargado.
    No requiere subir nada — usa el archivo que ya existe en videos_demo/.
    """
    demos = _cargar_meta_demos()
    demo = next((d for d in demos if d["id"] == request.demo_id), None)

    if not demo:
        raise HTTPException(404, f"Video demo '{request.demo_id}' no encontrado")

    ruta_video = os.path.join(CARPETA_DEMOS, demo["archivo"])
    if not os.path.exists(ruta_video):
        raise HTTPException(404, f"El archivo del video demo no existe en disco")

    session_id = str(uuid.uuid4())

    SESIONES[session_id] = {
        "ruta": ruta_video,
        "nombre_original": demo["nombre"],
        "es_demo": True,
    }

    return {
        "session_id": session_id,
        "nombre_video": demo["nombre"],
        "video_url": f"/sesion/{session_id}/video",
        "intervalos_robo": demo.get("intervalos_robo", []),
    }


@app.get("/sesion/{session_id}/video")
def obtener_video(session_id: str):
    if session_id not in SESIONES:
        raise HTTPException(404, "Sesión no encontrada")

    ruta = SESIONES[session_id]["ruta"]
    if not os.path.exists(ruta):
        raise HTTPException(404, "Video no existe en disco")

    return FileResponse(ruta, media_type="video/mp4")

def agregar_metricas_a_resultado(resultado: Dict, intervalos_robo: List[IntervaloRobo]) -> Dict:
    """
    Recibe el resultado de un detector y le agrega métricas usando los intervalos reales.
    También elimina prediccion_frames antes de devolverlo al frontend.
    """
    prediccion_frames = resultado.get("prediccion_frames")
    fps_video = resultado.get("fps_video")

    if prediccion_frames is not None and fps_video is not None:
        intervalos = [
            (i.inicio_seg, i.fin_seg)
            for i in intervalos_robo
        ]

        metricas = calcular_metricas(
            prediccion_frames=prediccion_frames,
            fps=fps_video,
            intervalos=intervalos
        )

        resultado["metricas"] = metricas
        resultado.pop("prediccion_frames", None)
    else:
        resultado["metricas"] = None
        resultado["advertencia_metricas"] = (
            "Este modelo no devolvió prediccion_frames o fps_video, "
            "por eso no se pudieron calcular métricas."
        )

    return resultado

@app.post("/sesion/{session_id}/correr/{nombre_modelo}")
def correr_modelo(session_id: str, nombre_modelo: str, request: CorrerModeloRequest):
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
        resultado = agregar_metricas_a_resultado(resultado, request.intervalos_robo)
        return resultado

    except Exception as e:
        raise HTTPException(500, f"Error procesando: {str(e)}")

@app.get("/sesion/{session_id}/stream/{nombre_modelo}")
def stream_modelo_visual(session_id: str, nombre_modelo: str):
    """
    Devuelve un stream MJPEG con bounding boxes, etiquetas y HUD.
    Solo se usa para los 3 modelos Pose seleccionados.
    """
    if session_id not in SESIONES:
        raise HTTPException(404, "Sesión no encontrada")

    if nombre_modelo not in MODELOS_DISPONIBLES:
        raise HTTPException(404, f"Modelo '{nombre_modelo}' no disponible")

    if nombre_modelo not in MODELOS_VISUALES:
        raise HTTPException(
            400,
            f"El modelo '{nombre_modelo}' no tiene visualización en vivo habilitada"
        )

    ruta_video = SESIONES[session_id]["ruta"]
    if not os.path.exists(ruta_video):
        raise HTTPException(404, "Video no existe en disco")

    detector = MODELOS_DISPONIBLES[nombre_modelo]

    if not hasattr(detector, "procesar_video_streaming"):
        raise HTTPException(
            400,
            f"El modelo '{nombre_modelo}' no soporta streaming visual"
        )

    detector.ultimo_resultado = None

    return StreamingResponse(
        detector.procesar_video_streaming(ruta_video),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/sesion/{session_id}/resultado-stream/{nombre_modelo}")
def obtener_resultado_stream(session_id: str, nombre_modelo: str, request: CorrerModeloRequest):
    """
    El frontend consulta este endpoint mientras el stream está procesando.
    Cuando el stream termina, devuelve las métricas finales.
    """
    if session_id not in SESIONES:
        raise HTTPException(404, "Sesión no encontrada")

    if nombre_modelo not in MODELOS_DISPONIBLES:
        raise HTTPException(404, f"Modelo '{nombre_modelo}' no disponible")

    if nombre_modelo not in MODELOS_VISUALES:
        raise HTTPException(
            400,
            f"El modelo '{nombre_modelo}' no tiene resultado visual habilitado"
        )

    detector = MODELOS_DISPONIBLES[nombre_modelo]
    resultado = getattr(detector, "ultimo_resultado", None)

    if resultado is None:
        return JSONResponse(
            status_code=202,
            content={"estado": "procesando"}
        )

    resultado = dict(resultado)
    resultado = agregar_metricas_a_resultado(resultado, request.intervalos_robo)
    return resultado

@app.delete("/sesion/{session_id}")
def cerrar_sesion(session_id: str):
    """
    Cierra la sesión. Si era de un video subido, lo borra del disco.
    Si era de un demo, NO borra nada (los demos son permanentes).
    """
    if session_id not in SESIONES:
        return {"mensaje": "Sesión ya no existe"}

    info = SESIONES[session_id]
    ruta = info["ruta"]
    es_demo = info.get("es_demo", False)

    del SESIONES[session_id]

    # Solo borrar el video si NO era un demo
    if not es_demo:
        time.sleep(0.3)
        try:
            if os.path.exists(ruta):
                os.remove(ruta)
        except PermissionError:
            print(f"[WARN] No se pudo borrar {ruta} ahora")

    return {"mensaje": "Sesión cerrada"}


# ============================================================
# STARTUP - LIMPIEZA
# ============================================================

@app.on_event("startup")
def limpiar_videos_huerfanos():
    """Al arrancar el servidor, borra videos subidos viejos. NO toca los demos."""
    if not os.path.exists(CARPETA_VIDEOS):
        return
    for archivo in os.listdir(CARPETA_VIDEOS):
        ruta = os.path.join(CARPETA_VIDEOS, archivo)
        try:
            os.remove(ruta)
        except Exception:
            pass
    print("[Startup] Carpeta de videos subidos limpiada")


# ============================================================
# CÁLCULO DE MÉTRICAS (sin cambios)
# ============================================================

def normalizar_prediccion(valor: str) -> str:
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
    total = len(prediccion_frames)

    if total == 0 or fps <= 0:
        return {
            "accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0,
            "matriz_confusion": {"TP": 0, "TN": 0, "FP": 0, "FN": 0},
            "mensaje": "No hay frames o FPS inválido."
        }

    verdad_frames = []
    for i in range(total):
        tiempo = i / fps
        es_robo = any(inicio <= tiempo <= fin for inicio, fin in intervalos)
        verdad_frames.append("SHOPLIFTING" if es_robo else "Normal")

    prediccion_frames = [normalizar_prediccion(p) for p in prediccion_frames]

    TP = sum(1 for p, v in zip(prediccion_frames, verdad_frames) if p == "SHOPLIFTING" and v == "SHOPLIFTING")
    TN = sum(1 for p, v in zip(prediccion_frames, verdad_frames) if p == "Normal" and v == "Normal")
    FP = sum(1 for p, v in zip(prediccion_frames, verdad_frames) if p == "SHOPLIFTING" and v == "Normal")
    FN = sum(1 for p, v in zip(prediccion_frames, verdad_frames) if p == "Normal" and v == "SHOPLIFTING")

    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "matriz_confusion": {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
        "total_frames": total,
        "fps_video": fps,
        "intervalos_evaluados": intervalos,
    }