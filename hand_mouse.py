"""Control del mouse del sistema con la mano.

Usa MediaPipe GestureRecognizer (API de Tasks) para:

* localizar las manos (21 landmarks cada una) y elegir la más prominente
  (mayor área en imagen = más cerca/visible, con histéresis para no
  saltar entre manos);
* mover el cursor siguiendo la MUÑECA de esa mano — se usa la muñeca y
  no la punta de los dedos porque la muñeca no se desplaza al cerrar el
  puño, evitando que el cursor salte justo al hacer click;
* detectar el gesto "Closed_Fist" (puño cerrado) y disparar UN click
  izquierdo por cierre, con tiempo mínimo de confirmación y cooldown.

El frame que recibe ya está en modo espejo, así que mover la mano a la
derecha mueve el cursor a la derecha de forma natural.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from pynput.mouse import Button, Controller

from pose_detector import download_model

GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)

# Zona activa del frame (fracciones del ancho/alto) que se mapea a TODA la
# pantalla: así no hace falta estirar la mano hasta los bordes de la cámara.
ACTIVE_X = (0.15, 0.85)
ACTIVE_Y = (0.15, 0.80)

CURSOR_SMOOTHING = 0.4    # EMA del cursor: más alto = más reactivo, menos filtrado
FIST_HOLD_SECONDS = 0.15  # el puño debe mantenerse este tiempo para confirmar el click
CLICK_COOLDOWN = 1.0      # segundos mínimos entre dos clicks
HAND_SWITCH_RATIO = 1.4   # la otra mano debe verse 40 % más grande para robar el control
MIN_GESTURE_SCORE = 0.5

_GESTURE_NAMES_ES = {
    "Closed_Fist": "Puño",
    "Open_Palm": "Mano abierta",
    "Pointing_Up": "Apuntando",
    "Victory": "Victoria",
    "Thumb_Up": "Pulgar arriba",
    "Thumb_Down": "Pulgar abajo",
    "ILoveYou": "ILY",
    "None": "—",
}


@dataclass(frozen=True)
class HandMouseState:
    """Estado por frame del control de mouse por mano (para la GUI y overlay)."""

    active: bool = False
    hand_label: str = ""    # "Derecha" / "Izquierda"
    gesture: str = "—"      # nombre del gesto en español
    fist: bool = False
    clicked: bool = False   # True solo en el frame donde se disparó el click
    pointer: tuple[float, float] | None = None  # muñeca en coords normalizadas


@dataclass(frozen=True)
class _Hand:
    """Datos mínimos de una mano detectada en un frame."""

    label: str
    wrist: tuple[float, float]
    area: float
    gesture: str
    score: float


def _map_range(value: float, low: float, high: float) -> float:
    """Reubica ``value`` de [low, high] a [0, 1], con recorte en los bordes."""
    return min(1.0, max(0.0, (value - low) / (high - low)))


class HandMouseController:
    """Sigue la mano más visible con el cursor y hace click al cerrar el puño."""

    def __init__(self, screen_size: tuple[int, int]) -> None:
        # Import perezoso por la misma razón que en pose_detector: se crea
        # dentro del hilo de video sin congelar la GUI.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(
                model_asset_path=download_model(GESTURE_MODEL_URL)
            ),
            running_mode=vision.RunningMode.VIDEO,  # tracking entre frames
            num_hands=2,  # se detectan ambas manos y luego se elige la mejor
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

        self._screen_w, self._screen_h = screen_size
        self._mouse = Controller()
        # La GUI puede apagar el control sin destruir el objeto; el tracking
        # sigue activo para mostrar el estado en pantalla.
        self.enabled = True

        # El modo VIDEO exige timestamps en ms estrictamente crecientes.
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1

        self._smoothed: tuple[float, float] | None = None
        self._current_hand: str | None = None  # etiqueta de la mano con el control
        self._fist_since: float | None = None
        self._last_click = 0.0
        self._armed = True  # exige abrir la mano entre un click y el siguiente

    # ------------------------------------------------------------------

    def update(self, frame_bgr: np.ndarray, now: float) -> HandMouseState:
        """Procesa un frame (ya espejado y sin overlays) y actúa sobre el mouse."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int((now - self._start_time) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        output = self._recognizer.recognize_for_video(mp_image, timestamp_ms)
        hands = self._extract_hands(output)
        if not hands:
            # Sin mano: se libera el cursor y se olvida el estado del gesto.
            self._smoothed = None
            self._current_hand = None
            self._fist_since = None
            self._armed = True
            return HandMouseState()

        hand = self._choose_hand(hands)

        # --- Cursor: zona activa -> pantalla completa, con suavizado EMA ---
        nx = _map_range(hand.wrist[0], *ACTIVE_X)
        ny = _map_range(hand.wrist[1], *ACTIVE_Y)
        if self._smoothed is None:
            self._smoothed = (nx, ny)
        else:
            a = CURSOR_SMOOTHING
            self._smoothed = (
                a * nx + (1 - a) * self._smoothed[0],
                a * ny + (1 - a) * self._smoothed[1],
            )
        if self.enabled:
            self._mouse.position = (
                int(self._smoothed[0] * self._screen_w),
                int(self._smoothed[1] * self._screen_h),
            )

        # --- Click por puño ------------------------------------------------
        fist = hand.gesture == "Closed_Fist" and hand.score >= MIN_GESTURE_SCORE
        clicked = self._update_click(fist, now)

        label = {"Right": "Derecha", "Left": "Izquierda"}.get(hand.label, "Mano")
        return HandMouseState(
            active=True,
            hand_label=label,
            gesture=_GESTURE_NAMES_ES.get(hand.gesture, hand.gesture),
            fist=fist,
            clicked=clicked,
            pointer=hand.wrist,
        )

    def close(self) -> None:
        self._recognizer.close()

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_hands(output) -> list[_Hand]:
        """Convierte la salida de MediaPipe en una lista plana de _Hand."""
        hands: list[_Hand] = []
        for i, landmarks in enumerate(output.hand_landmarks):
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            # Área del rectángulo que encierra la mano: proxy de "qué tanto
            # se ve" (una mano más cercana ocupa más imagen).
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            label = output.handedness[i][0].category_name if output.handedness else ""
            if output.gestures and output.gestures[i]:
                gesture = output.gestures[i][0].category_name
                score = output.gestures[i][0].score
            else:
                gesture, score = "None", 0.0
            hands.append(_Hand(label, (landmarks[0].x, landmarks[0].y), area, gesture, score))
        return hands

    def _choose_hand(self, hands: list[_Hand]) -> _Hand:
        """Elige la mano más grande, con histéresis para no cambiar por ruido."""
        best = max(hands, key=lambda h: h.area)
        if self._current_hand is not None and best.label != self._current_hand:
            current = next((h for h in hands if h.label == self._current_hand), None)
            # La mano actual conserva el control salvo que la otra sea
            # claramente más prominente.
            if current is not None and best.area < current.area * HAND_SWITCH_RATIO:
                best = current
        self._current_hand = best.label
        return best

    def _update_click(self, fist: bool, now: float) -> bool:
        """Máquina de estados del click: puño sostenido -> un solo click."""
        if not fist:
            self._fist_since = None
            self._armed = True  # mano abierta: se rearma para el próximo click
            return False
        if self._fist_since is None:
            self._fist_since = now
        held_enough = now - self._fist_since >= FIST_HOLD_SECONDS
        cooled_down = now - self._last_click >= CLICK_COOLDOWN
        if self._armed and held_enough and cooled_down:
            if self.enabled:
                self._mouse.click(Button.left)
            self._last_click = now
            self._armed = False
            return True
        return False


_MARKER_COLOR = (0, 255, 255)  # amarillo (BGR)


def draw_hand_marker(frame_bgr: np.ndarray, state: HandMouseState) -> None:
    """Anillo sobre la mano que controla el cursor; se rellena al cerrar el puño."""
    if not state.active or state.pointer is None:
        return
    height, width = frame_bgr.shape[:2]
    center = (int(state.pointer[0] * width), int(state.pointer[1] * height))
    if state.fist:
        cv2.circle(frame_bgr, center, 12, _MARKER_COLOR, -1, cv2.LINE_AA)
    else:
        cv2.circle(frame_bgr, center, 12, _MARKER_COLOR, 2, cv2.LINE_AA)
