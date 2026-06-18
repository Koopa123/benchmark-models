import time
from collections import Counter
from typing import Dict, List
from pathlib import Path

import cv2
from ultralytics import YOLO

from vision_core.modelos.base import DetectorBase


CLASES_NOMBRE = {0: "Normal", 1: "SHOPLIFTING"}
CONF_DETECCION = 0.4
IOU_THRESHOLD = 0.3
MAX_FRAMES_SIN_VER = 30


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


class YOLOv8BinarioDetector(DetectorBase):

    def __init__(self, nombre: str, ruta_pesos):
        """
        ruta_pesos puede ser un str o un Path. Se convierte a str para YOLO.
        """
        self.nombre = nombre
        self.ruta_pesos = str(ruta_pesos)
        self.model = None

    def cargar(self):
        print(f"[{self.nombre}] Cargando...")
        self.model = YOLO(self.ruta_pesos)
        print(f"[{self.nombre}] Listo.")

    def procesar_video(self, ruta_video: str) -> Dict:
        if self.model is None:
            self.cargar()

        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {ruta_video}")

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

        tracks_activos: List[Dict] = []
        tracks_cerrados: List[Dict] = []
        next_id = 1

        # Predicción por frame: "SHOPLIFTING" si detectó al menos una caja clase=1, sino "Normal"
        prediccion_frames: List[str] = []

        frame_idx = 0
        tiempo_inicio = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = self.model.predict(frame, conf=CONF_DETECCION, verbose=False)

            detecciones_frame = []
            hay_robo = False
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    clase_id = int(box.cls[0])
                    if clase_id == 1:
                        hay_robo = True
                    detecciones_frame.append({
                        "bbox": [float(c) for c in box.xyxy[0]],
                        "clase": clase_id,
                        "conf": float(box.conf[0]),
                    })

            prediccion_frames.append("SHOPLIFTING" if hay_robo else "Normal")

            # Tracking por IoU
            ids_asignados = set()
            for det in detecciones_frame:
                mejor_iou = 0.0
                mejor_track = None
                for t in tracks_activos:
                    if t["id"] in ids_asignados:
                        continue
                    iou_val = iou(det["bbox"], t["bbox"])
                    if iou_val > mejor_iou and iou_val >= IOU_THRESHOLD:
                        mejor_iou = iou_val
                        mejor_track = t

                if mejor_track is not None:
                    mejor_track["bbox"] = det["bbox"]
                    mejor_track["votos"].append(det["clase"])
                    mejor_track["confianzas"].append(det["conf"])
                    mejor_track["frames_sin_ver"] = 0
                    ids_asignados.add(mejor_track["id"])
                else:
                    tracks_activos.append({
                        "id": next_id,
                        "bbox": det["bbox"],
                        "votos": [det["clase"]],
                        "confianzas": [det["conf"]],
                        "frames_sin_ver": 0,
                    })
                    ids_asignados.add(next_id)
                    next_id += 1

            sobreviven = []
            for t in tracks_activos:
                if t["id"] not in ids_asignados:
                    t["frames_sin_ver"] += 1
                if t["frames_sin_ver"] >= MAX_FRAMES_SIN_VER:
                    tracks_cerrados.append(t)
                else:
                    sobreviven.append(t)
            tracks_activos = sobreviven

        cap.release()
        tiempo_total = time.time() - tiempo_inicio
        todos = tracks_activos + tracks_cerrados

        # Resumen por persona
        resultado_personas = []
        for t in todos:
            if not t["votos"]:
                continue
            contador = Counter(t["votos"])
            clase_ganadora = contador.most_common(1)[0][0]
            confianzas_ganadoras = [
                c for v, c in zip(t["votos"], t["confianzas"])
                if v == clase_ganadora
            ]
            confianza_prom = (
                sum(confianzas_ganadoras) / len(confianzas_ganadoras)
                if confianzas_ganadoras else 0.0
            )
            resultado_personas.append({
                "id": t["id"],
                "prediccion": CLASES_NOMBRE[clase_ganadora],
                "confianza": round(confianza_prom * 100, 2),
                "total_detecciones": len(t["votos"]),
            })

        return {
            "nombre_modelo": self.nombre,
            "tiempo_total_s": round(tiempo_total, 2),
            "tiempo_por_frame_ms": round((tiempo_total * 1000) / max(frame_idx, 1), 2),
            "fps_efectivo": round(frame_idx / tiempo_total, 2) if tiempo_total > 0 else 0,
            "fps_video": round(fps_video, 2),
            "total_frames": frame_idx,
            "personas_detectadas": resultado_personas,
            "prediccion_frames": prediccion_frames,
        }

    def liberar(self):
        del self.model
        self.model = None
