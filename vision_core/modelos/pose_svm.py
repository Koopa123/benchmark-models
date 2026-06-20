import joblib
import numpy as np

from vision_core.modelos.pose_base import PoseDetectorBase, normalize_pose_keypoints
from vision_core.paths import CLASIFICADOR_SVM, _str


class PoseSVMDetector(PoseDetectorBase):

    nombre = "YOLO Pose + SVM"

    def __init__(self):
        super().__init__(self.nombre)
        self.ruta_pesos = CLASIFICADOR_SVM

    def cargar_clasificador(self):
        self.clasificador = joblib.load(_str(self.ruta_pesos))

    def predecir(self, keypoints_xy, keypoints_xyn):
        # normalize_pose_keypoints siempre construye el dict en el MISMO
        # orden (es una función determinística), así que convertir sus
        # valores a un array numpy preserva el orden de columnas que
        # el modelo espera, sin pasar por pandas.
        features = normalize_pose_keypoints(keypoints_xy)
        valores = np.array([list(features.values())], dtype=np.float64)

        clase = int(self.clasificador.predict(valores)[0])

        if hasattr(self.clasificador, "predict_proba"):
            probas = self.clasificador.predict_proba(valores)[0]
            confianza = float(probas[clase])
        else:
            confianza = 1.0

        return clase, confianza