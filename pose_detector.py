"""Capa de detección de pose humana.

Este módulo aísla al resto de la aplicación del modelo concreto de pose.
La GUI solo conoce dos contratos:

* ``BasePoseDetector``: procesa un frame BGR y devuelve un ``PoseResult``
  (o ``None`` si no hay persona en la imagen).
* ``PoseResult``: contenedor genérico de landmarks normalizados.

Para sustituir MediaPipe por otro modelo (p. ej. YOLOv8-pose) basta con
implementar otra subclase de ``BasePoseDetector`` que devuelva un
``PoseResult`` equivalente (adaptando los índices de ``SKELETON_CONNECTIONS``
y ``LIMB_GROUPS`` al esquema de keypoints del nuevo modelo), sin tocar
la interfaz gráfica ni el rastreador de movimiento.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Índices de landmarks del esquema de MediaPipe Pose (33 puntos).
# Se definen como constantes propias para no acoplar el resto del código
# al paquete mediapipe.
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Umbral mínimo de "visibility" para considerar que un landmark es fiable.
VISIBILITY_THRESHOLD = 0.5

# Huesos del esqueleto que se dibujan sobre el video (pares de landmarks).
SKELETON_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Brazos
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    # Torso
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    # Piernas
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
)

# Agrupación de landmarks por extremidad. La GUI usa este diccionario para
# mostrar los indicadores de qué partes del cuerpo se están detectando.
# "Izquierdo/derecho" se refiere al lado anatómico REAL de la persona.
LIMB_GROUPS: dict[str, tuple[int, ...]] = {
    "Cabeza": (NOSE, LEFT_EAR, RIGHT_EAR),
    "Hombros": (LEFT_SHOULDER, RIGHT_SHOULDER),
    "Brazo izquierdo": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    "Brazo derecho": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
    "Torso": (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP),
    "Pierna izquierda": (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
    "Pierna derecha": (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
}


@dataclass
class PoseResult:
    """Resultado genérico de una detección de pose.

    ``landmarks`` es un array (N, 4) con columnas [x, y, z, visibility],
    donde x, y están normalizados al rango 0..1 respecto al frame.
    """

    landmarks: np.ndarray

    def flip_horizontal(self) -> "PoseResult":
        """Refleja las coordenadas X para mostrar la vista en espejo."""
        flipped = self.landmarks.copy()
        flipped[:, 0] = 1.0 - flipped[:, 0]
        return PoseResult(flipped)

    def limb_status(self, threshold: float = VISIBILITY_THRESHOLD) -> dict[str, bool]:
        """Devuelve qué extremidades tienen visibilidad media suficiente."""
        status: dict[str, bool] = {}
        for name, indices in LIMB_GROUPS.items():
            visibilities = self.landmarks[list(indices), 3]
            status[name] = bool(visibilities.mean() >= threshold)
        return status

    def center_point(self) -> tuple[float, float] | None:
        """Punto de referencia del cuerpo usado para el rastreo de movimiento.

        Prioridad: centro de hombros -> centro de caderas -> nariz.
        Se prefieren los hombros porque casi siempre son visibles y se
        mueven poco por sí solos (a diferencia de manos o pies), lo que da
        una señal de desplazamiento del cuerpo mucho más estable.
        """
        for a_idx, b_idx in ((LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP)):
            a, b = self.landmarks[a_idx], self.landmarks[b_idx]
            if a[3] >= VISIBILITY_THRESHOLD and b[3] >= VISIBILITY_THRESHOLD:
                return float((a[0] + b[0]) / 2), float((a[1] + b[1]) / 2)
        nose = self.landmarks[NOSE]
        if nose[3] >= VISIBILITY_THRESHOLD:
            return float(nose[0]), float(nose[1])
        return None


class BasePoseDetector(ABC):
    """Contrato mínimo que debe cumplir cualquier detector de pose."""

    @abstractmethod
    def process(self, frame_bgr: np.ndarray) -> PoseResult | None:
        """Procesa un frame BGR y devuelve la pose, o None si no hay persona."""

    def close(self) -> None:
        """Libera los recursos del modelo (opcional)."""


class MediaPipePoseDetector(BasePoseDetector):
    """Detector basado en MediaPipe Pose (BlazePose, 33 landmarks).

    ``model_complexity``: 0 = lite (máximo FPS), 1 = full (equilibrado),
    2 = heavy (más preciso pero lento en CPU).
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # Import perezoso: mediapipe tarda varios segundos en cargar, así que
        # solo se importa al crear el detector (dentro del hilo de video),
        # evitando congelar la GUI al arrancar la aplicación.
        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,  # modo video: reutiliza tracking entre frames
            model_complexity=model_complexity,
            enable_segmentation=False,  # no se necesita la máscara de segmentación
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> PoseResult | None:
        # MediaPipe espera RGB; OpenCV entrega BGR.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False  # permite a MediaPipe procesar sin copiar
        output = self._pose.process(rgb)
        if not output.pose_landmarks:
            return None
        data = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in output.pose_landmarks.landmark],
            dtype=np.float32,
        )
        return PoseResult(data)

    def close(self) -> None:
        self._pose.close()


# ---------------------------------------------------------------------------
# Dibujo del esqueleto (independiente del modelo: solo usa PoseResult).
# ---------------------------------------------------------------------------

# Articulaciones que se dibujan; se omiten los puntos finos de la cara
# (ojos, boca...) para no saturar el overlay.
_DRAWN_POINTS: tuple[int, ...] = tuple(
    sorted({i for a, b in SKELETON_CONNECTIONS for i in (a, b)} | {NOSE})
)

_BONE_COLOR = (80, 220, 120)    # verde (BGR)
_JOINT_FILL = (255, 255, 255)   # blanco
_JOINT_RING = (60, 60, 200)     # rojo oscuro


def draw_skeleton(
    frame_bgr: np.ndarray,
    result: PoseResult,
    threshold: float = VISIBILITY_THRESHOLD,
) -> None:
    """Dibuja huesos (líneas) y articulaciones (círculos) sobre el frame.

    Solo se dibujan los segmentos cuyos dos extremos superan el umbral de
    visibilidad, para no pintar extremidades que el modelo está adivinando.
    """
    height, width = frame_bgr.shape[:2]
    landmarks = result.landmarks
    # Conversión de coordenadas normalizadas (0..1) a píxeles.
    pixels = (landmarks[:, :2] * np.array([width, height])).astype(int)

    for a, b in SKELETON_CONNECTIONS:
        if landmarks[a, 3] >= threshold and landmarks[b, 3] >= threshold:
            cv2.line(frame_bgr, tuple(pixels[a]), tuple(pixels[b]), _BONE_COLOR, 2, cv2.LINE_AA)

    for idx in _DRAWN_POINTS:
        if landmarks[idx, 3] >= threshold:
            center = tuple(pixels[idx])
            cv2.circle(frame_bgr, center, 5, _JOINT_RING, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, center, 3, _JOINT_FILL, -1, cv2.LINE_AA)
