"""
Detector de aglomeraciones para realtime.

Reusa la lógica del backend principal pero simplificada:
- Sin presets, sin zonas ignoradas, sin BD
- Stateless: cada frame se procesa de forma independiente
- Usa YOLOv8n (más rápido) en vez de YOLOv8s (más preciso)
"""
import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from vision_core.paths import YOLO_TRACKING, _str


# ============================================================
# CONFIGURACIÓN
# ============================================================

CONFIANZA_MINIMA = 0.40   # un poco más permisivo que el de pregrabado

DISTANCIA_AGRUPACION_FALLBACK = 100
FACTOR_DISTANCIA_AGRUPACION = 1.5

UMBRAL_MEDIO = 4   # de 2 a 3 personas → BAJO, 4-5 → MEDIO
UMBRAL_ALTO = 6    # 6 o más → ALTO

# Colores BGR (igual que el detector de pregrabado)
COLOR_BAJO = (0, 255, 0)        # verde
COLOR_MEDIO = (0, 255, 255)     # amarillo
COLOR_ALTO = (0, 0, 255)        # rojo
COLOR_HUD_BG = (30, 30, 30)
COLOR_HUD_TEXT = (255, 255, 255)


# ============================================================
# DETECTOR
# ============================================================

class AglomeracionDetector:
    """
    Detector de aglomeraciones. Procesa un frame a la vez (stateless).
    """

    nombre = "Detección de Aglomeraciones"

    def __init__(self):
        self.yolo = None

    def cargar(self):
        print(f"[{self.nombre}] Cargando...")
        # YOLO_TRACKING ya apunta a yolov8n.pt, perfecto para realtime
        self.yolo = YOLO(_str(YOLO_TRACKING))
        print(f"[{self.nombre}] Listo.")

    # ============================================================
    # API PRINCIPAL: procesar un frame de realtime
    # ============================================================
    def procesar_frame_realtime(self, frame, fps_actual: float = 0.0) -> Tuple[np.ndarray, List[Dict], Dict]:
        """
        Procesa un frame y devuelve (frame_anotado, detecciones, stats).

        stats contiene:
            - total_personas: int
            - grupo_mayor: int
            - nivel: "BAJO" | "MEDIO" | "ALTO"
        """
        if self.yolo is None:
            self.cargar()

        personas = self._detectar_personas(frame)
        distancia_umbral = self._calcular_distancia_adaptativa(personas)
        grupos = self._agrupar_personas(personas, distancia_umbral)
        grupo_mayor = self._obtener_grupo_mas_grande(grupos)
        nivel, color = self._clasificar(grupo_mayor)

        # Dibujar
        self._dibujar_personas(frame, personas)
        self._dibujar_grupos(frame, personas, grupos, distancia_umbral)
        self._dibujar_hud(frame, fps_actual, len(personas), grupo_mayor, nivel, color)

        # Detecciones en formato común para el WebSocket
        detecciones = [
            {
                "bbox": list(p["bbox"]),
                "conf": p["confianza"],
            }
            for p in personas
        ]

        stats = {
            "total_personas": len(personas),
            "grupo_mayor": grupo_mayor,
            "nivel": nivel,
        }

        return frame, detecciones, stats

    # ============================================================
    # DETECCIÓN DE PERSONAS (YOLOv8n)
    # ============================================================
    def _detectar_personas(self, frame) -> List[Dict]:
        # imgsz=480 acelera mucho la inferencia en CPU
        # (a costa de un poco de precisión, pero para realtime conviene)
        resultados = self.yolo(frame, imgsz=480, verbose=False)
        personas = []

        for resultado in resultados:
            if resultado.boxes is None:
                continue
            for caja in resultado.boxes:
                clase = int(caja.cls[0])
                confianza = float(caja.conf[0])

                # Clase 0 en COCO = persona
                if clase == 0 and confianza >= CONFIANZA_MINIMA:
                    x1, y1, x2, y2 = map(int, caja.xyxy[0])
                    centro_x = int((x1 + x2) / 2)
                    centro_y = int((y1 + y2) / 2)
                    ancho = x2 - x1

                    personas.append({
                        "bbox": (x1, y1, x2, y2),
                        "centro": (centro_x, centro_y),
                        "confianza": confianza,
                        "ancho": ancho,
                    })

        return personas

    # ============================================================
    # AGRUPACIÓN
    # ============================================================
    @staticmethod
    def _calcular_distancia(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _calcular_distancia_adaptativa(self, personas) -> float:
        if len(personas) < 2:
            return DISTANCIA_AGRUPACION_FALLBACK

        anchos = [p["ancho"] for p in personas if p["ancho"] > 0]
        if not anchos:
            return DISTANCIA_AGRUPACION_FALLBACK

        ancho_promedio = sum(anchos) / len(anchos)
        return ancho_promedio * FACTOR_DISTANCIA_AGRUPACION

    def _agrupar_personas(self, personas, distancia_umbral) -> List[List[int]]:
        grupos = []
        visitados = set()

        for i in range(len(personas)):
            if i in visitados:
                continue

            grupo_actual = []
            cola = [i]
            visitados.add(i)

            while cola:
                idx = cola.pop(0)
                grupo_actual.append(idx)

                for j in range(len(personas)):
                    if j not in visitados:
                        d = self._calcular_distancia(
                            personas[idx]["centro"],
                            personas[j]["centro"]
                        )
                        if d <= distancia_umbral:
                            visitados.add(j)
                            cola.append(j)

            grupos.append(grupo_actual)

        return grupos

    @staticmethod
    def _obtener_grupo_mas_grande(grupos) -> int:
        if not grupos:
            return 0
        return max(len(g) for g in grupos)

    # ============================================================
    # CLASIFICACIÓN
    # ============================================================
    @staticmethod
    def _clasificar(grupo_mayor: int) -> Tuple[str, Tuple[int, int, int]]:
        if grupo_mayor < UMBRAL_MEDIO:
            return "BAJO", COLOR_BAJO
        elif grupo_mayor < UMBRAL_ALTO:
            return "MEDIO", COLOR_MEDIO
        else:
            return "ALTO", COLOR_ALTO

    # ============================================================
    # DIBUJO
    # ============================================================
    @staticmethod
    def _dibujar_personas(frame, personas):
        for p in personas:
            x1, y1, x2, y2 = p["bbox"]
            cx, cy = p["centro"]
            conf = p["confianza"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 80), 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            texto = f"{conf * 100:.0f}%"
            cv2.putText(frame, texto, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 80), 1)

    def _dibujar_grupos(self, frame, personas, grupos, distancia_umbral):
        for grupo in grupos:
            if len(grupo) <= 1:
                continue
            puntos = [personas[i]["centro"] for i in grupo]
            for i in range(len(puntos)):
                for j in range(i + 1, len(puntos)):
                    d = self._calcular_distancia(puntos[i], puntos[j])
                    if d <= distancia_umbral:
                        cv2.line(frame, puntos[i], puntos[j], (255, 255, 0), 1)

    @staticmethod
    def _dibujar_hud(frame, fps, total, grupo_mayor, nivel, color):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        cv2.putText(frame, "Aglomeraciones", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUD_TEXT, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}  |  Personas: {total}  |  Grupo: {grupo_mayor}",
                    (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_HUD_TEXT, 1)

        # Nivel en la esquina derecha
        texto_nivel = f"NIVEL: {nivel}"
        (tw, th), _ = cv2.getTextSize(texto_nivel, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(frame, texto_nivel, (w - tw - 12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if nivel == "ALTO":
            cv2.putText(frame, "ALERTA: AGLOMERACION DETECTADA",
                        (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ALTO, 2)