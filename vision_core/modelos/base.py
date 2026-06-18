from abc import ABC, abstractmethod
from typing import Dict


class DetectorBase(ABC):
    """Interfaz base para todos los detectores del benchmark."""

    nombre: str = "BaseDetector"

    @abstractmethod
    def cargar(self) -> None:
        """Carga el modelo en memoria. Se llama una sola vez."""
        pass

    @abstractmethod
    def procesar_video(self, ruta_video: str) -> Dict:
        """Procesa un video y devuelve métricas + predicciones por persona."""
        pass

    def liberar(self) -> None:
        """Libera memoria del modelo."""
        pass