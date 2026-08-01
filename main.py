"""Punto de entrada de la aplicación.

Lanza la interfaz gráfica (PySide6). Toda la lógica vive en los demás
módulos:

* pose_detector.py  -> modelo de pose (MediaPipe, intercambiable)
* motion_tracker.py -> cálculo de dirección/velocidad de movimiento
* gui.py            -> ventana principal + hilo de video
"""

import sys

from PySide6.QtWidgets import QApplication

from gui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Detector de Humanos")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
