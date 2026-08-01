"""Rastreador de movimiento: dirección y velocidad entre frames.

Idea central
------------
En cada frame se recibe el "punto de referencia" del cuerpo (por defecto
el punto medio entre los hombros) en coordenadas NORMALIZADAS (0..1,
independientes de la resolución). Para decidir hacia dónde se mueve la
persona se aplican tres pasos:

1. **Suavizado (EMA)**: la pose vibra frame a frame por el ruido propio
   del modelo; un promedio móvil exponencial filtra esa vibración antes
   de medir desplazamientos.
2. **Ventana temporal**: se guardan los puntos suavizados de los últimos
   ~0.4 s y la velocidad se calcula como
   ``(punto_reciente - punto_antiguo) / Δt``. Medir sobre una ventana
   (y no entre 2 frames consecutivos) hace el resultado estable aunque
   el FPS fluctúe.
3. **Zona muerta**: si |velocidad| no supera un umbral, la persona se
   considera quieta. Esto evita que el indicador parpadee
   izquierda/derecha por micro-movimientos involuntarios.

Las velocidades se expresan en "fracción del frame por segundo":
vx = 0.10 significa desplazarse el 10 % del ancho de la imagen en 1 s.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

# Flechas usadas para construir la etiqueta legible de dirección.
_ARROWS = {"izquierda": "←", "derecha": "→", "arriba": "↑", "abajo": "↓"}


@dataclass(frozen=True)
class MotionState:
    """Estado de movimiento calculado para un frame."""

    vx: float = 0.0            # velocidad horizontal (fracción de frame / s)
    vy: float = 0.0            # velocidad vertical   (fracción de frame / s)
    speed: float = 0.0         # magnitud de la velocidad
    horizontal: str = ""       # "izquierda" | "derecha" | ""
    vertical: str = ""         # "arriba" | "abajo" | ""
    label: str = "Sin persona" # texto listo para mostrar en la GUI
    tracking: bool = False     # True si hay una persona siendo rastreada


def _build_label(horizontal: str, vertical: str) -> str:
    """Construye la etiqueta visible, p. ej. "→↑ Derecha + Arriba"."""
    parts = [d for d in (horizontal, vertical) if d]
    if not parts:
        return "Quieto"
    arrows = "".join(_ARROWS[d] for d in parts)
    return f"{arrows} {' + '.join(p.capitalize() for p in parts)}"


class MotionTracker:
    """Calcula dirección y velocidad del cuerpo a partir de un punto de referencia.

    Parámetros
    ----------
    window_seconds:
        Tamaño de la ventana temporal sobre la que se mide la velocidad.
    smoothing_alpha:
        Peso del EMA (0..1). Más alto = responde más rápido pero filtra menos.
    threshold_x / threshold_y:
        Zona muerta: velocidad mínima (fracción de frame por segundo) para
        considerar que hay movimiento real. El umbral vertical es menor
        porque los desplazamientos verticales del torso suelen ser cortos
        (agacharse, saltar) comparados con caminar lateralmente.
    """

    def __init__(
        self,
        window_seconds: float = 0.4,
        smoothing_alpha: float = 0.5,
        threshold_x: float = 0.10,
        threshold_y: float = 0.08,
    ) -> None:
        self._window = window_seconds
        self._alpha = smoothing_alpha
        self._threshold_x = threshold_x
        self._threshold_y = threshold_y
        # Historial de puntos suavizados: tuplas (timestamp, x, y).
        self._history: deque[tuple[float, float, float]] = deque()
        self._smoothed: tuple[float, float] | None = None

    def reset(self) -> None:
        """Olvida el historial (al perder a la persona o cambiar de cámara)."""
        self._history.clear()
        self._smoothed = None

    def update(self, point: tuple[float, float] | None, timestamp: float) -> MotionState:
        """Registra el punto del frame actual y devuelve el estado de movimiento.

        ``point`` debe venir en coordenadas normalizadas (0..1) y ya en
        orientación de pantalla (vista espejo), de modo que "derecha"
        signifique moverse hacia la derecha de la pantalla.
        """
        # Sin persona: se reinicia el historial para no mezclar trayectorias
        # de detecciones distintas (evita falsos saltos de dirección).
        if point is None:
            self.reset()
            return MotionState()

        # --- 1) Suavizado exponencial (EMA) -----------------------------
        if self._smoothed is None:
            self._smoothed = point
        else:
            a = self._alpha
            self._smoothed = (
                a * point[0] + (1 - a) * self._smoothed[0],
                a * point[1] + (1 - a) * self._smoothed[1],
            )
        x, y = self._smoothed

        # --- 2) Ventana temporal ----------------------------------------
        self._history.append((timestamp, x, y))
        while self._history and timestamp - self._history[0][0] > self._window:
            self._history.popleft()

        t0, x0, y0 = self._history[0]
        dt = timestamp - t0
        if dt < 0.05:
            # Todavía no hay suficiente historial para una velocidad fiable.
            return MotionState(label="Detectando…", tracking=True)

        vx = (x - x0) / dt
        vy = (y - y0) / dt
        speed = math.hypot(vx, vy)

        # --- 3) Clasificación con zona muerta ---------------------------
        horizontal = ""
        if vx <= -self._threshold_x:
            horizontal = "izquierda"
        elif vx >= self._threshold_x:
            horizontal = "derecha"

        # OJO: en coordenadas de imagen el eje Y crece hacia ABAJO, por eso
        # una velocidad vy negativa significa que la persona sube.
        vertical = ""
        if vy <= -self._threshold_y:
            vertical = "arriba"
        elif vy >= self._threshold_y:
            vertical = "abajo"

        return MotionState(
            vx=vx,
            vy=vy,
            speed=speed,
            horizontal=horizontal,
            vertical=vertical,
            label=_build_label(horizontal, vertical),
            tracking=True,
        )
