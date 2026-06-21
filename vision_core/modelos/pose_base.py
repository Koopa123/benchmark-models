import time
from collections import Counter
from typing import Dict, List, Generator, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from vision_core.modelos.base import DetectorBase
from vision_core.paths import YOLO_POSE_GENERICO, _str


CLASES = {0: "SHOPLIFTING", 1: "Normal"}
CONF_DETECCION_PERSONA = 0.5
IOU_THRESHOLD = 0.3
MAX_FRAMES_SIN_VER = 30

# Colores para dibujo (BGR porque es OpenCV)
COLOR_NORMAL = (0, 200, 80)
COLOR_SOSPECHOSO = (0, 50, 220)
COLOR_HUD_BG = (30, 30, 30)
COLOR_HUD_TEXT = (255, 255, 255)


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


def dibujar_hud(frame, nombre_modelo, frame_idx, total_frames, sospechosos):
    """Dibuja un panel informativo arriba del frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, nombre_modelo, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD_TEXT, 2)

    if total_frames > 0:
        progreso = f"Frame {frame_idx} / {total_frames}"
    else:
        progreso = f"Frame {frame_idx}"
    cv2.putText(frame, progreso, (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD_TEXT, 1)

    if sospechosos > 0:
        texto = f"SOSPECHOSOS: {sospechosos}"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(frame, texto, (w - tw - 12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)

    return frame


def dibujar_hud_realtime(frame, nombre_modelo, fps_actual, total_personas, sospechosos):
    """HUD para modo realtime (sin contador de frames totales, con FPS)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, nombre_modelo, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD_TEXT, 2)
    cv2.putText(frame, f"FPS: {fps_actual:.1f}  |  Personas: {total_personas}", (12, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD_TEXT, 1)

    if sospechosos > 0:
        texto = f"ALERTA: {sospechosos} sospechoso(s)"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(frame, texto, (w - tw - 12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)

    return frame


def dibujar_cajas(frame, detecciones):
    """Dibuja las cajas y etiquetas de cada detección."""
    for det in detecciones:
        x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
        es_sospechoso = det["clase"] == 0  # 0 = SHOPLIFTING
        color = COLOR_SOSPECHOSO if es_sospechoso else COLOR_NORMAL
        label = "SOSPECHOSO" if es_sospechoso else "NORMAL"
        conf = det.get("conf", 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        texto = f"{label} {conf * 100:.0f}%"
        (text_w, text_h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 10, y1), color, -1)
        cv2.putText(frame, texto, (x1 + 5, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


class PoseDetectorBase(DetectorBase):

    # Cada subclase puede sobreescribir esto si usa un YOLO Pose distinto
    ruta_yolo_pose = YOLO_POSE_GENERICO

    def __init__(self, nombre: str):
        self.nombre = nombre
        self.yolo_pose = None
        self.clasificador = None
        # Cache del último resultado para que el endpoint lo recupere
        # después de que termine el stream
        self.ultimo_resultado = None

    def cargar(self):
        print(f"[{self.nombre}] Cargando...")
        self.yolo_pose = YOLO(_str(self.ruta_yolo_pose))
        self.cargar_clasificador()
        print(f"[{self.nombre}] Listo.")

    def cargar_clasificador(self):
        raise NotImplementedError

    def predecir(self, keypoints_xy, keypoints_xyn):
        raise NotImplementedError

    # ============================================================
    # PROCESAR VIDEO (modo "silencioso", igual a antes)
    # ============================================================
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
        predicciones_por_frame: List[List[Dict]] = []

        frame_idx = 0
        tiempo_inicio = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            detecciones_frame = self._procesar_un_frame(frame)
            predicciones_por_frame.append(detecciones_frame)

            tracks_activos, tracks_cerrados, next_id = self._actualizar_tracks(
                tracks_activos, tracks_cerrados, detecciones_frame, next_id
            )

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

    # ============================================================
    # PROCESAR VIDEO STREAMING (modo "visual" para benchmark)
    # ============================================================
    def procesar_video_streaming(self, ruta_video: str) -> Generator[bytes, None, None]:
        """
        Versión generador: procesa el video frame por frame y yieldea
        cada frame procesado como bytes JPEG (multipart MJPEG).
        Al terminar, guarda el resultado en self.ultimo_resultado.
        """
        if self.yolo_pose is None:
            self.cargar()

        cap = cv2.VideoCapture(ruta_video)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {ruta_video}")

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Tiempo objetivo entre frames para reproducir al ritmo del video
        # (ej: 30 FPS → 0.0333s por frame). Sin esto, con GPU rápida el
        # video pasa "fast forward" y no se aprecia la detección.
        frame_duracion_objetivo = 1.0 / fps_video if fps_video > 0 else 0.0

        tracks_activos: List[Dict] = []
        tracks_cerrados: List[Dict] = []
        next_id = 1
        predicciones_por_frame: List[List[Dict]] = []

        frame_idx = 0
        tiempo_inicio = time.time()

        try:
            while True:
                inicio_frame = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                detecciones_frame = self._procesar_un_frame(frame)
                predicciones_por_frame.append(detecciones_frame)

                tracks_activos, tracks_cerrados, next_id = self._actualizar_tracks(
                    tracks_activos, tracks_cerrados, detecciones_frame, next_id
                )

                sospechosos_frame = sum(1 for d in detecciones_frame if d["clase"] == 0)
                frame = dibujar_cajas(frame, detecciones_frame)
                frame = dibujar_hud(
                    frame,
                    self.nombre,
                    frame_idx,
                    total_frames_video,
                    sospechosos_frame
                )

                ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue

                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame_bytes +
                    b"\r\n"
                )

                # Mantener el ritmo del video original.
                # Si el procesado fue más rápido que 1/fps_video, dormimos
                # el tiempo restante. Si fue más lento, seguimos sin esperar.
                tiempo_usado = time.time() - inicio_frame
                sleep_restante = frame_duracion_objetivo - tiempo_usado
                if sleep_restante > 0:
                    time.sleep(sleep_restante)
        finally:
            cap.release()
            tiempo_total = time.time() - tiempo_inicio
            todos_los_tracks = tracks_activos + tracks_cerrados

            self.ultimo_resultado = self._resultado_final(
                todos_los_tracks,
                predicciones_por_frame,
                frame_idx,
                tiempo_total,
                fps_video
            )
            print(f"[{self.nombre}] Stream terminado: {frame_idx} frames en {tiempo_total:.1f}s")

    # ============================================================
    # PROCESAR FRAME REALTIME (modo "vivo" desde WebSocket)
    # ============================================================
    def procesar_frame_realtime(self, frame, fps_actual: float = 0.0) -> Tuple[np.ndarray, List[Dict]]:
        """
        Procesa un solo frame (en BGR) y devuelve (frame_anotado, detecciones).
        Pensado para realtime: NO mantiene tracks ni guarda historial.

        Returns:
            (frame_anotado, lista_detecciones)
            lista_detecciones: [{"bbox": [x1,y1,x2,y2], "clase": 0|1, "conf": float, "es_sospechoso": bool}, ...]
        """
        if self.yolo_pose is None:
            self.cargar()

        # IMPORTANTE: usábamos imgsz=320 en CPU para ir más rápido, pero a esa
        # resolución los keypoints quedan muy pegados y los modelos que
        # dependen de normalize_pose_keypoints (SVM, XGB-Norm) generan features
        # mal escaladas, lo que produce predicciones triviales (todo "Normal")
        # y por eso visualmente parecía que "no detectaban nada".
        # En GPU (RTX 4090) imgsz=640 corre igual de rápido y da features sanas.
        detecciones = self._procesar_un_frame(frame, imgsz=640)


        # Enriquecer detecciones con flag es_sospechoso para el frontend
        for d in detecciones:
            d["es_sospechoso"] = d["clase"] == 0

        # Dibujar visualmente
        sospechosos = sum(1 for d in detecciones if d["clase"] == 0)
        frame = dibujar_cajas(frame, detecciones)
        frame = dibujar_hud_realtime(
            frame, self.nombre, fps_actual, len(detecciones), sospechosos
        )

        return frame, detecciones

    # ============================================================
    # HELPERS INTERNOS
    # ============================================================
    def _procesar_un_frame(self, frame, imgsz: int = None) -> List[Dict]:
        """Procesa un solo frame y devuelve las detecciones."""
        kwargs = {"verbose": False}
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        results = self.yolo_pose(frame, **kwargs)
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
                except Exception as e:
                    # Antes este except silenciaba TODO error en silencio,
                    # lo que ocultaba bugs de SVM/XGB-Norm. Ahora al menos
                    # imprime el error la primera vez para poder diagnosticar.
                    if not getattr(self, "_error_predecir_mostrado", False):
                        print(f"[{self.nombre}] WARN error en predecir(): {e}")
                        self._error_predecir_mostrado = True
                    continue

                # --- DEBUG TEMPORAL: borrar después de diagnosticar ---
                print(f"[{self.nombre}] DETECCION -> clase={clase_id} conf={confianza:.3f} bbox={bbox}")
                # --- FIN DEBUG TEMPORAL ---

                detecciones_frame.append({
                    "bbox": bbox,
                    "clase": clase_id,
                    "conf": confianza,
                })

        return detecciones_frame

    def _actualizar_tracks(self, tracks_activos, tracks_cerrados, detecciones_frame, next_id):
        """Actualiza los tracks por IoU. Devuelve (activos, cerrados, next_id)."""
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

        return sobreviven, tracks_cerrados, next_id

    def _resultado_final(self, tracks, predicciones_por_frame, frame_idx, tiempo_total, fps_video):
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
