import time
from collections import Counter
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import YOLO

from modelos.base import DetectorBase


CLASES = {0: "SHOPLIFTING", 1: "Normal"}
CONF_DETECCION_PERSONA = 0.5
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


class PoseDetectorBase(DetectorBase):

    ruta_yolo_pose: str = "pesos/yolov8n-pose.pt"

    def __init__(self, nombre: str):
        self.nombre = nombre
        self.yolo_pose = None
        self.clasificador = None

    def cargar(self):
        print(f"[{self.nombre}] Cargando...")
        self.yolo_pose = YOLO(self.ruta_yolo_pose)
        self.cargar_clasificador()
        print(f"[{self.nombre}] Listo.")

    def cargar_clasificador(self):
        raise NotImplementedError

    def predecir(self, keypoints_xy, keypoints_xyn):
        raise NotImplementedError

    def procesar_video(self, ruta_video: str) -> Dict:
        if self.yolo_pose is None:
            self.cargar()

        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {ruta_video}")

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0

        tracks_activos: List[Dict] = []
        tracks_cerrados: List[Dict] = []
        next_id = 1

        # Predicciones frame-a-frame
        # Para cada frame: lista de {bbox, clase, conf}
        predicciones_por_frame: List[List[Dict]] = []

        frame_idx = 0
        tiempo_inicio = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            results = self.yolo_pose(frame, verbose=False)
            detecciones_frame = []

            for r in results:
                if r.keypoints is None or r.boxes is None:
                    continue
                boxes = r.boxes.xyxy
                confs = r.boxes.conf.tolist()
                kpts_xy = r.keypoints.xy.cpu().numpy()
                kpts_xyn = r.keypoints.xyn.cpu().numpy()

                for idx, box in enumerate(boxes):
                    if confs[idx] < CONF_DETECCION_PERSONA:
                        continue

                    bbox = [float(c) for c in box.tolist()]

                    try:
                        clase_id, confianza = self.predecir(kpts_xy[idx], kpts_xyn[idx])
                    except Exception:
                        continue

                    detecciones_frame.append({
                        "bbox": bbox,
                        "clase": clase_id,
                        "conf": confianza,
                    })

            # Guardar predicciones del frame
            predicciones_por_frame.append(detecciones_frame)

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
        todos_los_tracks = tracks_activos + tracks_cerrados

        return self._resultado_final(
            todos_los_tracks,
            predicciones_por_frame,
            frame_idx,
            tiempo_total,
            fps_video
        )

    def _resultado_final(self, tracks, predicciones_por_frame, frame_idx, tiempo_total, fps_video):
        # Resumen por persona (lo que ya teníamos)
        resultado_personas = []
        for t in tracks:
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
                "prediccion": CLASES[clase_ganadora],
                "confianza": round(confianza_prom * 100, 2),
                "total_detecciones": len(t["votos"]),
            })

        # Predicción por frame: si HAY al menos una detección con clase=0 (SHOPLIFTING),
        # el frame se marca como SHOPLIFTING
        prediccion_frames = []
        for dets in predicciones_por_frame:
            if any(d["clase"] == 0 for d in dets):
                prediccion_frames.append("SHOPLIFTING")
            else:
                prediccion_frames.append("Normal")

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
        del self.yolo_pose, self.clasificador
        self.yolo_pose = self.clasificador = None


# ============================================================
# UTILIDADES DE FEATURES (sin cambios)
# ============================================================

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
KP = {name: i for i, name in enumerate(KEYPOINT_NAMES)}


def normalize_pose_keypoints(keypoints_xy):
    keypoints = np.asarray(keypoints_xy, dtype=float)

    left_shoulder = keypoints[KP["left_shoulder"]]
    right_shoulder = keypoints[KP["right_shoulder"]]
    left_hip = keypoints[KP["left_hip"]]
    right_hip = keypoints[KP["right_hip"]]

    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0
    body_center = (shoulder_mid + hip_mid) / 2.0

    torso_scale = np.linalg.norm(shoulder_mid - hip_mid)
    shoulder_scale = np.linalg.norm(left_shoulder - right_shoulder)
    scale = torso_scale if torso_scale > 1 else shoulder_scale
    if scale <= 1:
        scale = 1.0

    features = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        x, y = keypoints[i]
        features[f"{name}_x_norm"] = (x - body_center[0]) / scale
        features[f"{name}_y_norm"] = (y - body_center[1]) / scale

    pairs = [
        ("left_wrist", "left_hip"),
        ("right_wrist", "right_hip"),
        ("left_wrist", "right_hip"),
        ("right_wrist", "left_hip"),
        ("left_wrist", "left_shoulder"),
        ("right_wrist", "right_shoulder"),
        ("left_wrist", "right_wrist"),
        ("left_elbow", "left_hip"),
        ("right_elbow", "right_hip"),
    ]

    for a, b in pairs:
        pa = keypoints[KP[a]]
        pb = keypoints[KP[b]]
        features[f"dist_{a}_to_{b}"] = np.linalg.norm(pa - pb) / scale

    features["torso_dx"] = (shoulder_mid[0] - hip_mid[0]) / scale
    features["torso_dy"] = (shoulder_mid[1] - hip_mid[1]) / scale

    return features