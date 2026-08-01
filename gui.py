"""Interfaz gráfica del detector de humanos.

Arquitectura
------------
* ``VideoWorker`` (QThread): captura frames de la cámara, ejecuta el
  detector de pose y el rastreador de movimiento SIN bloquear la GUI.
  Comunica los resultados al hilo principal mediante señales Qt, que son
  thread-safe.
* ``MainWindow``: ventana principal. Recibe frames ya procesados y solo
  se encarga de pintarlos y refrescar los indicadores (FPS, dirección,
  extremidades).

El worker crea su propio detector/tracker dentro de ``run()`` para que
tanto la carga de MediaPipe como el procesamiento ocurran íntegramente
en el hilo secundario.
"""

from __future__ import annotations

import sys
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent, QGuiApplication, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hand_mouse import HandMouseController, draw_hand_marker
from motion_tracker import MotionTracker
from pose_detector import (
    LIMB_GROUPS,
    MediaPipePoseDetector,
    draw_bounding_box,
    draw_skeleton,
)

# Resolución solicitada a la cámara: 640x480 es el mejor equilibrio
# precisión/FPS para MediaPipe corriendo en CPU.
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480

# Complejidad del modelo de pose: 0 = más FPS, 1 = equilibrado, 2 = más preciso.
MODEL_COMPLEXITY = 1

# Cuántos índices de cámara se prueban al buscar dispositivos.
MAX_CAMERAS_TO_PROBE = 5

APP_STYLE = """
QMainWindow, QWidget {
    background-color: #14161f;
    color: #e6e6e6;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QLabel#videoLabel {
    background-color: #000000;
    border: 1px solid #2a2d3a;
    border-radius: 4px;
    color: #7a7d8c;
    font-size: 16px;
}
QLabel#sectionTitle {
    color: #8ab4ff;
    font-weight: bold;
    margin-top: 10px;
}
QLabel#directionLabel {
    font-size: 24px;
    font-weight: bold;
    color: #ffd166;
}
QLabel#fpsLabel {
    font-size: 15px;
    color: #9ad1a5;
}
QPushButton {
    background-color: #2f6fdb;
    color: white;
    border: none;
    padding: 8px 12px;
    border-radius: 4px;
    font-weight: bold;
}
QPushButton:hover { background-color: #3d7ff0; }
QPushButton:disabled { background-color: #3a3d4d; color: #808080; }
QComboBox {
    background-color: #232633;
    border: 1px solid #3a3d4d;
    padding: 4px;
    border-radius: 4px;
}
"""


def preferred_backend() -> int:
    """Backend de captura: DirectShow abre mucho más rápido en Windows."""
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def list_cameras(max_devices: int = MAX_CAMERAS_TO_PROBE) -> list[int]:
    """Prueba los índices 0..N-1 y devuelve los que abren una cámara real."""
    available: list[int] = []
    for index in range(max_devices):
        cap = cv2.VideoCapture(index, preferred_backend())
        if cap.isOpened():
            available.append(index)
        cap.release()
    return available


class VideoWorker(QThread):
    """Hilo de captura + procesamiento. Emite un frame ya dibujado por ciclo."""

    # frame BGR con overlay + diccionario de metadatos (fps, motion, limbs...)
    frame_ready = Signal(object, object)
    # error irrecuperable (cámara inexistente, desconexión, fallo del modelo)
    failed = Signal(str)

    def __init__(
        self,
        camera_index: int,
        screen_size: tuple[int, int],
        mouse_enabled: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._camera_index = camera_index
        self._screen_size = screen_size
        self._mouse_enabled = mouse_enabled

    def set_mouse_enabled(self, enabled: bool) -> None:
        # bool atómico (GIL): seguro de escribir desde el hilo de la GUI.
        self._mouse_enabled = enabled

    def run(self) -> None:  # se ejecuta en el hilo secundario
        cap = cv2.VideoCapture(self._camera_index, preferred_backend())
        if not cap.isOpened():
            self.failed.emit(f"No se pudo abrir la cámara {self._camera_index}.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            detector = MediaPipePoseDetector(model_complexity=MODEL_COMPLEXITY)
            hand_mouse = HandMouseController(self._screen_size)
        except Exception as exc:  # p. ej. falta descargar un modelo sin internet
            cap.release()
            self.failed.emit(f"No se pudo inicializar el modelo:\n{exc}")
            return

        tracker = MotionTracker()
        no_limbs = dict.fromkeys(LIMB_GROUPS, False)
        fps = 0.0
        last_time = time.monotonic()

        try:
            while not self.isInterruptionRequested():
                ok, frame = cap.read()
                if not ok or frame is None:
                    # La cámara se desconectó o dejó de entregar frames.
                    self.failed.emit("Se perdió la señal de la cámara (¿desconectada?).")
                    return

                # 1) Detectar pose sobre el frame ORIGINAL (sin espejo) para
                #    que MediaPipe etiquete bien el lado izquierdo/derecho
                #    anatómico de la persona.
                result = detector.process(frame)

                # 2) Espejar el frame para mostrarlo como espejo (UX natural)
                #    y espejar también los landmarks para que el esqueleto y
                #    la dirección de movimiento coincidan con la pantalla.
                frame = cv2.flip(frame, 1)
                now = time.monotonic()

                # 3) Manos/mouse sobre el frame espejado LIMPIO (antes de
                #    dibujar overlays que taparian la mano).
                hand_mouse.enabled = self._mouse_enabled
                hand_state = hand_mouse.update(frame, now)

                if result is not None:
                    result = result.flip_horizontal()
                    draw_skeleton(frame, result)
                    draw_bounding_box(frame, result)
                    motion = tracker.update(result.center_point(), now)
                    limbs = result.limb_status()
                else:
                    motion = tracker.update(None, now)
                    limbs = no_limbs
                draw_hand_marker(frame, hand_state)

                # 3) FPS reales del pipeline (captura + inferencia + dibujo),
                #    suavizados con EMA para que no parpadeen.
                delta = now - last_time
                last_time = now
                if delta > 0:
                    instant = 1.0 / delta
                    fps = instant if fps == 0.0 else 0.9 * fps + 0.1 * instant

                meta = {
                    "fps": fps,
                    "motion": motion,
                    "limbs": limbs,
                    "person": result is not None,
                    "hand": hand_state,
                }
                self.frame_ready.emit(frame, meta)
        finally:
            cap.release()
            detector.close()
            hand_mouse.close()


class MainWindow(QMainWindow):
    """Ventana principal: video a la izquierda, panel de estado a la derecha."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Detector de Humanos — Pose y Movimiento en Tiempo Real")
        self.setStyleSheet(APP_STYLE)
        self._worker: VideoWorker | None = None
        self._limb_labels: dict[str, QLabel] = {}
        self._build_ui()
        self._refresh_cameras()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # --- Zona de video ------------------------------------------------
        self.video_label = QLabel("Cámara detenida")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(CAPTURE_WIDTH, CAPTURE_HEIGHT)
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        root.addWidget(self.video_label, stretch=1)

        # --- Panel lateral --------------------------------------------------
        panel = QVBoxLayout()
        panel.setSpacing(6)

        panel.addWidget(self._section_title("Cámara"))
        camera_row = QHBoxLayout()
        self.camera_combo = QComboBox()
        camera_row.addWidget(self.camera_combo, stretch=1)
        self.refresh_button = QPushButton("Buscar")
        self.refresh_button.setToolTip("Volver a detectar cámaras conectadas")
        self.refresh_button.clicked.connect(self._refresh_cameras)
        camera_row.addWidget(self.refresh_button)
        panel.addLayout(camera_row)

        self.start_button = QPushButton("Iniciar detección")
        self.start_button.clicked.connect(self._toggle_detection)
        panel.addWidget(self.start_button)

        self.mouse_checkbox = QCheckBox("Controlar mouse con la mano")
        self.mouse_checkbox.setChecked(True)
        self.mouse_checkbox.setToolTip(
            "El cursor sigue tu muñeca; cerrar el puño hace click izquierdo.\n"
            "Baja la mano para liberar el cursor."
        )
        self.mouse_checkbox.toggled.connect(self._on_mouse_toggle)
        panel.addWidget(self.mouse_checkbox)

        panel.addWidget(self._separator())

        panel.addWidget(self._section_title("Rendimiento"))
        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setObjectName("fpsLabel")
        panel.addWidget(self.fps_label)

        panel.addWidget(self._section_title("Movimiento"))
        self.direction_label = QLabel("—")
        self.direction_label.setObjectName("directionLabel")
        self.direction_label.setWordWrap(True)
        panel.addWidget(self.direction_label)

        panel.addWidget(self._section_title("Control por mano"))
        self.hand_label = QLabel("Sin mano")
        panel.addWidget(self.hand_label)

        panel.addWidget(self._section_title("Extremidades detectadas"))
        for limb_name in LIMB_GROUPS:
            label = QLabel(f"● {limb_name}")
            self._set_limb_state(label, detected=False)
            self._limb_labels[limb_name] = label
            panel.addWidget(label)

        panel.addStretch(1)

        side = QWidget()
        side.setLayout(panel)
        side.setFixedWidth(250)
        root.addWidget(side)

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("sectionTitle")
        return label

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    @staticmethod
    def _set_limb_state(label: QLabel, detected: bool) -> None:
        color = "#2ecc71" if detected else "#5c5f6e"
        label.setStyleSheet(f"color: {color};")

    # ------------------------------------------------------------- Cámaras

    @Slot()
    def _refresh_cameras(self) -> None:
        """Rellena el combo con las cámaras disponibles en este momento."""
        self.camera_combo.clear()
        cameras = list_cameras()
        if not cameras:
            self.camera_combo.addItem("Sin cámaras detectadas", -1)
            self.start_button.setEnabled(False)
        else:
            for index in cameras:
                self.camera_combo.addItem(f"Cámara {index}", index)
            self.start_button.setEnabled(True)

    # ------------------------------------------------------ Iniciar / parar

    @Slot()
    def _toggle_detection(self) -> None:
        if self._worker is not None:
            self._stop_detection()
        else:
            self._start_detection()

    def _start_detection(self) -> None:
        camera_index = self.camera_combo.currentData()
        if camera_index is None or camera_index < 0:
            return
        # pynput trabaja en píxeles físicos; Qt reporta lógicos, se corrige
        # con el factor de escala (DPI) del monitor principal.
        screen = QGuiApplication.primaryScreen()
        ratio = screen.devicePixelRatio()
        screen_size = (
            int(screen.geometry().width() * ratio),
            int(screen.geometry().height() * ratio),
        )
        self._worker = VideoWorker(
            camera_index, screen_size, self.mouse_checkbox.isChecked()
        )
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

        self.start_button.setText("Detener detección")
        self.start_button.setStyleSheet("background-color: #c0392b;")
        self.camera_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.video_label.setText("Iniciando cámara y modelo…")

    def _stop_detection(self) -> None:
        if self._worker is not None:
            # Petición cooperativa: el bucle del worker la revisa cada frame.
            self._worker.requestInterruption()
            self._worker.wait(3000)

    # --------------------------------------------------------------- Slots

    @Slot(object, object)
    def _on_frame(self, frame: np.ndarray, meta: dict) -> None:
        # Puede llegar un frame rezagado después de detener; se ignora.
        if self._worker is None:
            return

        # numpy BGR -> QPixmap (QPixmap.fromImage copia los datos, por lo
        # que el buffer numpy puede liberarse sin problema).
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        self.fps_label.setText(f"{meta['fps']:.1f} FPS")
        self.direction_label.setText(meta["motion"].label)

        hand = meta["hand"]
        if hand.active:
            text = f"{hand.hand_label}: {hand.gesture}"
            if hand.clicked:
                text += " — ¡CLICK!"
            self.hand_label.setText(text)
        else:
            self.hand_label.setText("Sin mano")

        for limb_name, detected in meta["limbs"].items():
            label = self._limb_labels.get(limb_name)
            if label is not None:
                self._set_limb_state(label, detected)

    @Slot(bool)
    def _on_mouse_toggle(self, checked: bool) -> None:
        if self._worker is not None:
            self._worker.set_mouse_enabled(checked)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        QMessageBox.warning(self, "Error de cámara", message)

    @Slot()
    def _on_worker_finished(self) -> None:
        """Restaura la UI cuando el hilo de video termina (por parada o error)."""
        self._worker = None
        self.start_button.setText("Iniciar detección")
        self.start_button.setStyleSheet("")
        self.camera_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.video_label.setText("Cámara detenida")
        self.fps_label.setText("0.0 FPS")
        self.direction_label.setText("—")
        self.hand_label.setText("Sin mano")
        for label in self._limb_labels.values():
            self._set_limb_state(label, detected=False)

    # --------------------------------------------------------------- Cierre

    def closeEvent(self, event: QCloseEvent) -> None:
        self._stop_detection()
        event.accept()
