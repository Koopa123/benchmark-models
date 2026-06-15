import time
from collections import deque
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from ultralytics import YOLO

from modelos.base import DetectorBase


NUM_FRAMES_T = 16
FRAMES_BUFFER = 16
RECLASIFICAR = 16
THRESHOLD_TRACK = 0.5


class PersonaTracker:
    def __init__(self, persona_id: int):
        self.id = persona_id
        self.frames = deque(maxlen=FRAMES_BUFFER)
        self.estado = "Analizando"
        self.prob_shoplifting = 0.0
        self.frames_desde_clasificacion = 0

    def agregar_frame(self, crop):
        frame_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        self.frames.append(frame_resized)
        self.frames_desde_clasificacion += 1

    def listo_para_clasificar(self) -> bool:
        return (
            len(self.frames) >= NUM_FRAMES_T
            and self.frames_desde_clasificacion >= RECLASIFICAR
        )

    def clasificar(self, model, processor, device):
        frames_list = list(self.frames)
        indices = np.linspace(0, len(frames_list) - 1, NUM_FRAMES_T).astype(int)
        frames_sel = [frames_list[i] for i in indices]

        inputs = processor(frames_sel, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            probs = F.softmax(outputs.logits, dim=-1)

        self.prob_shoplifting = probs[0][1].item()
        self.estado = "SHOPLIFTING" if self.prob_shoplifting > THRESHOLD_TRACK else "Normal"
        self.frames_desde_clasificacion = 0


class VideoMAEDetector(DetectorBase):

    nombre = "VideoMAE"

    def __init__(self, ruta_pesos: str = "pesos/modelo_videomae"):
        self.ruta_pesos = ruta_pesos
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.yolo = None

    def cargar(self):
        print(f"[{self.nombre}] Cargando en {self.device}...")
        self.yolo = YOLO("yolov8n.pt")
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.model = VideoMAEForVideoClassification.from_pretrained(self.ruta_pesos)
        self.model.to(self.device)
        self.model.eval()
        print(f"[{self.nombre}] Listo.")

    def procesar_video(self, ruta_video: str) -> Dict:
        if self.model is None:
            self.cargar()

        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {ruta_video}")

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        personas: Dict[int, PersonaTracker] = {}
        prediccion_frames: List[str] = []

        frame_idx = 0
        tiempo_inicio = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = self.yolo.track(frame, persist=True, classes=[0], verbose=False)
            hay_sospecha = False

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids = results[0].boxes.id.cpu().numpy().astype(int)

                for bbox, pid in zip(boxes, ids):
                    x1 = max(0, int(bbox[0]))
                    y1 = max(0, int(bbox[1]))
                    x2 = min(width, int(bbox[2]))
                    y2 = min(height, int(bbox[3]))
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue

                    if pid not in personas:
                        personas[pid] = PersonaTracker(pid)
                    personas[pid].agregar_frame(frame[y1:y2, x1:x2])

                    if personas[pid].listo_para_clasificar():
                        personas[pid].clasificar(self.model, self.processor, self.device)

                    # Si alguna persona en este frame está marcada como SHOPLIFTING, el frame lo es
                    if personas[pid].estado == "SHOPLIFTING":
                        hay_sospecha = True

            prediccion_frames.append("SHOPLIFTING" if hay_sospecha else "Normal")

        cap.release()
        tiempo_total = time.time() - tiempo_inicio

        resultado_personas = []
        for pid, p in personas.items():
            resultado_personas.append({
                "id": int(pid),
                "prediccion": p.estado if p.estado != "Analizando" else "Normal",
                "confianza": round(p.prob_shoplifting * 100, 2),
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
        del self.model, self.processor, self.yolo
        self.model = self.processor = self.yolo = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()