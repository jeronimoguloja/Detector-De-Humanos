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

import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

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

    def bounding_box(
        self, threshold: float = VISIBILITY_THRESHOLD, margin: float = 0.06
    ) -> tuple[float, float, float, float] | None:
        """Caja normalizada (x0, y0, x1, y1) que engloba los landmarks visibles."""
        visible = self.landmarks[self.landmarks[:, 3] >= threshold]
        if len(visible) < 2:
            return None
        return (
            max(0.0, float(visible[:, 0].min()) - margin),
            max(0.0, float(visible[:, 1].min()) - margin),
            min(1.0, float(visible[:, 0].max()) + margin),
            min(1.0, float(visible[:, 1].max()) + margin),
        )


class BasePoseDetector(ABC):
    """Contrato mínimo que debe cumplir cualquier detector de pose."""

    @abstractmethod
    def process(self, frame_bgr: np.ndarray) -> PoseResult | None:
        """Procesa un frame BGR y devuelve la pose, o None si no hay persona."""

    def close(self) -> None:
        """Libera los recursos del modelo (opcional)."""


# Modelos oficiales de PoseLandmarker por nivel de complejidad.
# Se descargan una sola vez a la carpeta local models/.
_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def download_model(url: str) -> str:
    """Descarga un modelo .task a la carpeta local models/ si aún no existe.

    También la usa hand_mouse.py para el reconocedor de gestos de mano.
    """
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / url.rsplit("/", 1)[-1]
    if not model_path.exists():
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as exc:
            model_path.unlink(missing_ok=True)  # no dejar descargas a medias
            raise RuntimeError(
                "No se pudo descargar el modelo (se necesita internet "
                f"solo la primera vez): {exc}"
            ) from exc
    return str(model_path)


def _ensure_model(model_complexity: int) -> str:
    return download_model(_MODEL_URLS[model_complexity])


class MediaPipePoseDetector(BasePoseDetector):
    """Detector basado en MediaPipe Tasks PoseLandmarker (BlazePose, 33 landmarks).

    Nota: mediapipe >= 0.10.30 eliminó la antigua API ``mp.solutions``;
    esta implementación usa la API de Tasks, que requiere un archivo de
    modelo .task (se descarga automáticamente la primera vez).

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
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model(model_complexity)),
            running_mode=vision.RunningMode.VIDEO,  # modo video: usa tracking entre frames
            num_poses=1,  # una sola persona: más rápido y suficiente para esta app
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        # El modo VIDEO exige timestamps en ms estrictamente crecientes.
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1

    def process(self, frame_bgr: np.ndarray) -> PoseResult | None:
        # MediaPipe espera RGB; OpenCV entrega BGR.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        output = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not output.pose_landmarks:
            return None
        person = output.pose_landmarks[0]  # num_poses=1: solo hay una lista
        data = np.array(
            [
                [lm.x, lm.y, lm.z, 1.0 if lm.visibility is None else lm.visibility]
                for lm in person
            ],
            dtype=np.float32,
        )
        return PoseResult(data)

    def close(self) -> None:
        self._landmarker.close()


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


_BOX_COLOR = (0, 210, 90)  # verde (BGR)


def draw_bounding_box(
    frame_bgr: np.ndarray,
    result: PoseResult,
    threshold: float = VISIBILITY_THRESHOLD,
) -> None:
    """Rectángulo verde fino alrededor de la persona detectada."""
    box = result.bounding_box(threshold)
    if box is None:
        return
    height, width = frame_bgr.shape[:2]
    top_left = (int(box[0] * width), int(box[1] * height))
    bottom_right = (int(box[2] * width), int(box[3] * height))
    # Grosor 1: visible pero discreto, no tapa la imagen.
    cv2.rectangle(frame_bgr, top_left, bottom_right, _BOX_COLOR, 1, cv2.LINE_AA)
