import xgboost as xgb
import numpy as np

from vision_core.modelos.pose_base import PoseDetectorBase
from vision_core.paths import YOLO_POSE_BOOSTER, CLASIFICADOR_XGB_BOOSTER, _str


class PoseXGBBoosterDetector(PoseDetectorBase):

    nombre = "YOLO Pose + XGBoost Booster"

    # Este experimento usa su propio YOLO Pose entrenado, no el genérico
    ruta_yolo_pose = YOLO_POSE_BOOSTER

    def __init__(self):
        super().__init__(self.nombre)
        self.ruta_clasificador = CLASIFICADOR_XGB_BOOSTER
        self._nombres_columnas = None  # se calcula una sola vez, no en cada frame

    def cargar_clasificador(self):
        self.clasificador = xgb.Booster()
        self.clasificador.load_model(_str(self.ruta_clasificador))

    def predecir(self, keypoints_xy, keypoints_xyn):
        n = len(keypoints_xyn)

        # Calculamos los nombres de columnas (x0,y0,x1,y1,...) una sola vez
        # y los cacheamos, en vez de reconstruir el dict en cada frame.
        if self._nombres_columnas is None:
            nombres = []
            for j in range(n):
                nombres.append(f"x{j}")
                nombres.append(f"y{j}")
            self._nombres_columnas = nombres

        # Mismo orden que antes (x0,y0,x1,y1,...) pero sin pasar por pandas
        valores = np.empty((1, n * 2), dtype=np.float32)
        for j in range(n):
            valores[0, j * 2] = keypoints_xyn[j][0]
            valores[0, j * 2 + 1] = keypoints_xyn[j][1]

        # feature_names hace que XGBoost matchee por NOMBRE, no por posición,
        # así que el resultado es idéntico al de antes con pandas.
        dmatrix = xgb.DMatrix(valores, feature_names=self._nombres_columnas)
        pred = self.clasificador.predict(dmatrix)[0]
        clase = int(pred > 0.5)
        confianza = pred if clase == 1 else (1.0 - pred)
        return clase, float(confianza)