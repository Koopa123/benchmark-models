import time
from collections import deque, defaultdict
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from modelos.base import DetectorBase


# Configuración (igual al notebook)
NUM_FRAMES_FAST = 32
ALPHA = 4
IMG_SIZE = 224
THRESHOLD = 0.5
WINDOW_SECONDS = 2.0   # ventana de tiempo por persona

MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


class SlowFastDetector(DetectorBase):

    nombre = "SlowFast"

    def __init__(self, ruta_pesos: str = "pesos/slowfast_best.pt"):
        self.ruta_pesos = ruta_pesos
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.yolo = None

    def cargar(self):
        print(f"[{self.nombre}] Cargando en {self.device}...")

        # Cargar arquitectura SlowFast R50 (sin pesos pretrained, los nuestros los pisan)
        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "slowfast_r50",
            pretrained=False
        )

        # Reemplazar la cabeza de clasificación (400 clases -> 2 clases)
        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(in_features, 2)

        # Cargar pesos fine-tuned
        state_dict = torch.load(self.ruta_pesos, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # YOLO para tracking de personas
        self.yolo = YOLO("yolov8n.pt")

        print(f"[{self.nombre}] Listo.")

    def _predecir_persona(self, frames_persona):
        """Recibe una lista de crops BGR de una misma persona, devuelve (pred, prob_shop, prob_normal)."""
        if len(frames_persona) < NUM_FRAMES_FAST:
            return "Normal", 0.0, 100.0

        indices = np.linspace(0, len(frames_persona) - 1, NUM_FRAMES_FAST).astype(int)

        selected_frames = []
        for i in indices:
            crop = frames_persona[i]
            if crop is None or crop.size == 0:
                continue
            crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            selected_frames.append(crop_rgb)

        if len(selected_frames) < NUM_FRAMES_FAST:
            return "Normal", 0.0, 100.0

        frames = np.stack(selected_frames)  # (T, H, W, C)

        # Preprocesamiento
        t = torch.from_numpy(frames).float() / 255.0
        t = t.permute(3, 0, 1, 2)  # (C, T, H, W)

        mean = MEAN.to(t.device)
        std = STD.to(t.device)
        t = (t - mean) / std

        # Generar pathway slow (submuestreo de fast)
        slow_idx = torch.linspace(0, NUM_FRAMES_FAST - 1, NUM_FRAMES_FAST // ALPHA).long()
        slow = torch.index_select(t, 1, slow_idx).unsqueeze(0).to(self.device)
        fast = t.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model([slow, fast])
            probs = F.softmax(logits, dim=-1)

        prob_normal = probs[0][0].item() * 100
        prob_shoplifting = probs[0][1].item() * 100

        pred = "SHOPLIFTING" if probs[0][1].item() > THRESHOLD else "Normal"
        return pred, prob_shoplifting, prob_normal

    def procesar_video(self, ruta_video: str) -> Dict:
        if self.model is None:
            self.cargar()

        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {ruta_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_needed = max(NUM_FRAMES_FAST, int(fps * WINDOW_SECONDS))

        track_buffers = defaultdict(lambda: deque(maxlen=frames_needed))
        track_predictions = {}

        frame_idx = 0
        tiempo_inicio = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = self.yolo.track(
                frame,
                persist=True,
                classes=[0],
                tracker="bytetrack.yaml",
                verbose=False
            )

            if results[0].boxes is None or results[0].boxes.id is None:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue

                track_buffers[track_id].append(person_crop)

                # Cuando tenga suficientes frames, clasifica
                if len(track_buffers[track_id]) >= frames_needed:
                    pred, prob_shop, prob_normal = self._predecir_persona(
                        list(track_buffers[track_id])
                    )
                    track_predictions[track_id] = {
                        "pred": pred,
                        "prob_shop": prob_shop,
                        "prob_normal": prob_normal,
                    }

        cap.release()
        tiempo_total = time.time() - tiempo_inicio

        # Armar resultado
        resultado_personas = []
        for track_id, info in track_predictions.items():
            resultado_personas.append({
                "id": int(track_id),
                "prediccion": info["pred"],
                "confianza": round(info["prob_shop"], 2),
            })

        return {
            "nombre_modelo": self.nombre,
            "tiempo_total_s": round(tiempo_total, 2),
            "tiempo_por_frame_ms": round((tiempo_total * 1000) / max(frame_idx, 1), 2),
            "fps_efectivo": round(frame_idx / tiempo_total, 2) if tiempo_total > 0 else 0,
            "total_frames": frame_idx,
            "personas_detectadas": resultado_personas,
        }

    def liberar(self):
        del self.model, self.yolo
        self.model = self.yolo = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()