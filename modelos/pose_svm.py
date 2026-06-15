import joblib
import pandas as pd

from modelos.pose_base import PoseDetectorBase, normalize_pose_keypoints


class PoseSVMDetector(PoseDetectorBase):

    nombre = "YOLO Pose + SVM"

    def __init__(self, ruta_pesos: str = "pesos/clasificador_pose_svm_normalizado.pkl"):
        super().__init__(self.nombre)
        self.ruta_pesos = ruta_pesos

    def cargar_clasificador(self):
        self.clasificador = joblib.load(self.ruta_pesos)

    def predecir(self, keypoints_xy, keypoints_xyn):
        # Este modelo fue entrenado con features de normalización corporal
        features = normalize_pose_keypoints(keypoints_xy)
        df = pd.DataFrame(features, index=[0])
        clase = int(self.clasificador.predict(df)[0])

        if hasattr(self.clasificador, "predict_proba"):
            probas = self.clasificador.predict_proba(df)[0]
            confianza = float(probas[clase])
        else:
            confianza = 1.0

        return clase, confianza