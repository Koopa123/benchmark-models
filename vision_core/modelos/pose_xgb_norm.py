import joblib
import numpy as np

from vision_core.modelos.pose_base import PoseDetectorBase, normalize_pose_keypoints
from vision_core.paths import CLASIFICADOR_XGB_NORM, _str


class PoseXGBNormDetector(PoseDetectorBase):

    nombre = "YOLO Pose + XGBoost Normalizado"

    def __init__(self):
        super().__init__(self.nombre)
        self.ruta_pesos = CLASIFICADOR_XGB_NORM

    def cargar_clasificador(self):
        self.clasificador = joblib.load(_str(self.ruta_pesos))

    def predecir(self, keypoints_xy, keypoints_xyn):
        features = normalize_pose_keypoints(keypoints_xy)
        valores = np.array([list(features.values())], dtype=np.float64)

        clase = int(self.clasificador.predict(valores)[0])

        if hasattr(self.clasificador, "predict_proba"):
            probas = self.clasificador.predict_proba(valores)[0]
            confianza = float(probas[clase])
        else:
            confianza = 1.0

        return clase, confianza