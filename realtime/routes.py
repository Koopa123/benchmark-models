"""
Router de Realtime.

Detección en tiempo real usando la webcam del CLIENTE (navegador).
NO abre cámara local del servidor.

Flujo:
  1. Cliente abre WebSocket a /realtime/ws?modo=sospechosa
  2. Cliente envía frames JPEG como bytes
  3. Servidor procesa con el detector y devuelve JSON con:
       - frame_jpeg_base64: el frame con cajas y HUD
       - detecciones: lista de detecciones con bbox, clase, conf
       - sospechosos: cuántos sospechosos hay en este frame
  4. Cliente espera la respuesta y manda el siguiente frame
     (patrón request→response, auto-regulado)
"""
import time
import base64
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from vision_core.modelos.pose_xgb_booster import PoseXGBBoosterDetector


# ============================================================
# DETECTORES (singleton, se cargan una sola vez)
# ============================================================

# Solo cargamos el modelo ganador para "sospechosa".
# Si después quieres agregar "aglomeraciones", inicializa aquí su detector también.

_detector_sospechosa: Optional[PoseXGBBoosterDetector] = None


def get_detector(modo: str):
    """
    Devuelve el detector correspondiente al modo.
    Lo carga en memoria la primera vez que se pide.
    """
    global _detector_sospechosa

    if modo == "sospechosa":
        if _detector_sospechosa is None:
            print("[Realtime] Inicializando detector de actividad sospechosa...")
            _detector_sospechosa = PoseXGBBoosterDetector()
            _detector_sospechosa.cargar()
        return _detector_sospechosa

    # TODO: cuando agregues aglomeraciones:
    # elif modo == "aglomeraciones":
    #     ...

    raise ValueError(f"Modo desconocido: '{modo}'. Modos válidos: 'sospechosa'")


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(prefix="/realtime", tags=["realtime"])


@router.get("/")
def inicio():
    return {"mensaje": "Realtime API funcionando"}


@router.get("/modos")
def listar_modos():
    """Modos disponibles para que el frontend los muestre."""
    return {
        "modos": [
            {
                "id": "sospechosa",
                "nombre": "Actividad Sospechosa",
                "descripcion": "Detecta posibles robos con Pose + XGBoost Booster",
                "disponible": True,
            },
            {
                "id": "aglomeraciones",
                "nombre": "Aglomeraciones",
                "descripcion": "Detecta y cuenta personas para identificar aglomeraciones",
                "disponible": False,  # cambiar a True cuando lo implementes
            },
        ]
    }


# ============================================================
# WEBSOCKET PRINCIPAL
# ============================================================

@router.websocket("/ws")
async def realtime_ws(websocket: WebSocket, modo: str = "sospechosa"):
    """
    WebSocket para detección en tiempo real.

    Query params:
        modo: "sospechosa" (por ahora el único disponible)

    Protocolo:
        Cliente envía: bytes binarios = JPEG del frame de su webcam
        Servidor responde: JSON con frame anotado en base64 + metadata

    Para cambiar de modo: cerrar y reabrir el WebSocket con otro query param.
    """
    await websocket.accept()

    try:
        detector = get_detector(modo)
    except ValueError as e:
        await websocket.send_json({"error": str(e)})
        await websocket.close()
        return

    print(f"[Realtime] Cliente conectado en modo '{modo}'")

    # Para calcular FPS de procesamiento
    tiempo_anterior = time.time()
    fps_suavizado = 0.0

    try:
        while True:
            # Esperar el siguiente frame del cliente
            frame_bytes = await websocket.receive_bytes()

            # Decodificar JPEG a numpy array (BGR)
            arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({
                    "error": "Frame inválido (no se pudo decodificar)"
                })
                continue

            # Calcular FPS de procesamiento
            ahora = time.time()
            dt = ahora - tiempo_anterior
            if dt > 0:
                fps_actual = 1.0 / dt
                fps_suavizado = (
                    0.8 * fps_suavizado + 0.2 * fps_actual
                    if fps_suavizado > 0 else fps_actual
                )
            tiempo_anterior = ahora

            # Procesar el frame con el detector
            try:
                frame_anotado, detecciones = detector.procesar_frame_realtime(
                    frame, fps_actual=fps_suavizado
                )
            except Exception as e:
                print(f"[Realtime] Error procesando frame: {e}")
                await websocket.send_json({"error": f"Error procesando: {str(e)}"})
                continue

            # Encodear el frame procesado como JPEG y a base64
            ok, buffer = cv2.imencode(".jpg", frame_anotado, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            if not ok:
                await websocket.send_json({"error": "No se pudo encodear el frame"})
                continue

            frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

            # Resumen de detecciones para el frontend
            sospechosos = sum(1 for d in detecciones if d.get("es_sospechoso"))

            # Enviar respuesta
            await websocket.send_json({
                "frame_jpeg_base64": frame_b64,
                "detecciones": [
                    {
                        "bbox": d["bbox"],
                        "clase": d["clase"],
                        "conf": d["conf"],
                        "es_sospechoso": d["es_sospechoso"],
                    }
                    for d in detecciones
                ],
                "total_personas": len(detecciones),
                "sospechosos": sospechosos,
                "fps": round(fps_suavizado, 1),
            })

    except WebSocketDisconnect:
        print(f"[Realtime] Cliente desconectado (modo '{modo}')")
    except Exception as e:
        print(f"[Realtime] Error inesperado: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
