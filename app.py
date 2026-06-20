"""
Backend de Visión (Benchmark + Realtime).

Se ejecuta con:
    uvicorn app:app --host 0.0.0.0 --port 8001

Rutas:
    /bench/*       → comparación de modelos contra videos
    /realtime/ws   → WebSocket de detección en vivo desde webcam del cliente
"""
import os
import torch
import cv2

# ============================================================
# CONFIGURACIÓN DE THREADS
# ============================================================
# Dejamos 1 núcleo libre para el sistema operativo, uvicorn,
# nginx, fail2ban, etc. El resto se lo damos a torch/cv2 para
# acelerar la inferencia de los modelos.
NUM_CORES_TOTAL = os.cpu_count() or 6
NUM_CORES_MODELOS = max(NUM_CORES_TOTAL - 1, 1)

torch.set_num_threads(NUM_CORES_MODELOS)
cv2.setNumThreads(NUM_CORES_MODELOS)

print(f"[Config] CPU total: {NUM_CORES_TOTAL} núcleos | Asignados a modelos: {NUM_CORES_MODELOS}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bench.routes import router as bench_router, limpiar_videos_huerfanos
from realtime.routes import router as realtime_router


app = FastAPI(title="Backend de Visión - Benchmark + Realtime")

# CORS abierto por ahora. En producción, restringir a tu dominio de Vercel.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar routers
app.include_router(bench_router)
app.include_router(realtime_router)


@app.get("/")
def inicio():
    return {
        "mensaje": "Backend de Visión funcionando",
        "endpoints": {
            "benchmark": "/bench/*",
            "realtime_ws": "/realtime/ws?modo=sospechosa",
            "realtime_modos": "/realtime/modos",
        }
    }


@app.on_event("startup")
def on_startup():
    limpiar_videos_huerfanos()
    print("[Startup] Backend de Visión listo")