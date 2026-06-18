"""
Rutas absolutas a los pesos de los modelos.

Calculadas desde la ubicación de este archivo, NO desde el CWD,
para que funcione sin importar desde dónde se arranque uvicorn.
"""
from pathlib import Path

# vision_core/
VISION_CORE_DIR = Path(__file__).parent.resolve()

# vision_core/pesos/
PESOS_DIR = VISION_CORE_DIR / "pesos"

# ============================================================
# RUTAS INDIVIDUALES POR MODELO
# ============================================================

# YOLOv8 Pose genérico (usado por pose_svm y pose_xgb_norm)
YOLO_POSE_GENERICO = PESOS_DIR / "yolov8n-pose.pt"

# Pose + XGBoost Booster (tiene su propio YOLO Pose entrenado)
YOLO_POSE_BOOSTER = PESOS_DIR / "pose_xgb_booster" / "pose_booster.pt"
CLASIFICADOR_XGB_BOOSTER = PESOS_DIR / "pose_xgb_booster" / "model_weights.json"

# Pose + SVM
CLASIFICADOR_SVM = PESOS_DIR / "clasificador_pose_svm_normalizado.pkl"

# Pose + XGBoost Normalizado
CLASIFICADOR_XGB_NORM = PESOS_DIR / "clasificador_pose_xgboost_normalizado.pkl"

# YOLOv8 binarios (experimentos)
YOLO_EXP5 = PESOS_DIR / "exp5_best.pt"
YOLO_EXP8 = PESOS_DIR / "exp8_best.pt"

# VideoMAE
VIDEOMAE_DIR = PESOS_DIR / "modelo_videomae"

# YOLOv8n para tracking en VideoMAE
# Si no existe localmente, ultralytics lo descarga automáticamente
YOLO_TRACKING = PESOS_DIR / "yolov8n.pt"


def _str(path: Path) -> str:
    """Convierte Path a string. Útil cuando una librería no acepta Path."""
    return str(path)
