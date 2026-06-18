import xgboost as xgb
import pandas as pd

from vision_core.modelos.pose_base import PoseDetectorBase
from vision_core.paths import YOLO_POSE_BOOSTER, CLASIFICADOR_XGB_BOOSTER, _str


class PoseXGBBoosterDetector(PoseDetectorBase):

    nombre = "YOLO Pose + XGBoost Booster"

    # Este experimento usa su propio YOLO Pose entrenado, no el genérico
    ruta_yolo_pose = YOLO_POSE_BOOSTER

    def __init__(self):
        super().__init__(self.nombre)
        self.ruta_clasificador = CLASIFICADOR_XGB_BOOSTER

    def cargar_clasificador(self):
        self.clasificador = xgb.Booster()
        self.clasificador.load_model(_str(self.ruta_clasificador))

    def predecir(self, keypoints_xy, keypoints_xyn):
        # Este modelo usa keypoints normalizados [0,1]: x0, y0, x1, y1, ...
        data = {}
        for j in range(len(keypoints_xyn)):
            data[f"x{j}"] = keypoints_xyn[j][0]
            data[f"y{j}"] = keypoints_xyn[j][1]

        df = pd.DataFrame(data, index=[0])
        dmatrix = xgb.DMatrix(df)
        pred = self.clasificador.predict(dmatrix)[0]
        clase = int(pred > 0.5)
        confianza = pred if clase == 1 else (1.0 - pred)
        return clase, float(confianza)
