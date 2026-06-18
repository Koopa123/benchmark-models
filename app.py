"""
Backend de Visión (Benchmark + Realtime).

Se ejecuta con:
    uvicorn app:app --host 0.0.0.0 --port 8001

Rutas:
    /bench/*       → comparación de modelos contra videos
    /realtime/ws   → WebSocket de detección en vivo desde webcam del cliente
"""
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
