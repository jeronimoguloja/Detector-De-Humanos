# Detector de Humanos — Pose y Movimiento en Tiempo Real

Sistema de detección y seguimiento de personas con cámara web: dibuja el
esqueleto (pose estimation) sobre el video en vivo, indica qué
extremidades se detectan y hacia qué dirección se mueve la persona, todo
desde una interfaz gráfica.

## Características

- Detección de pose (33 landmarks) con **MediaPipe Pose** (BlazePose),
  optimizada para correr fluida en CPU.
- Overlay del esqueleto (articulaciones y huesos) sobre el video en
  vista espejo.
- Dirección de movimiento (izquierda / derecha / arriba / abajo)
  calculada a partir del desplazamiento del centro de los hombros.
- Indicadores por extremidad: cabeza, hombros, brazos, torso y piernas.
- FPS del pipeline en tiempo real.
- Selección de cámara (si hay varias) e inicio/paro desde la GUI.

## Requisitos

- **Python 3.10 – 3.12** (recomendado 3.11; verifica que exista una
  rueda de `mediapipe` para tu versión exacta de Python).
- Una cámara web.
- No se necesita GPU.

## Instalación

```powershell
cd "d:\UDEM\Modelos IA\Detector de humanos"
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> En Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`.

## Uso

```powershell
python main.py
```

1. Elige la cámara en el panel derecho (botón **Buscar** para volver a
   detectar dispositivos).
2. Pulsa **Iniciar detección**. La primera vez tarda unos segundos
   mientras se carga el modelo.
3. El panel derecho muestra FPS, la dirección de movimiento y qué
   extremidades se están detectando (verde = detectada).
4. Pulsa **Detener detección** o cierra la ventana para terminar.

## Estructura del proyecto

| Archivo             | Rol                                                                 |
| ------------------- | ------------------------------------------------------------------- |
| `main.py`           | Punto de entrada; lanza la GUI.                                      |
| `gui.py`            | Ventana principal (PySide6) + `VideoWorker` (QThread de captura).    |
| `pose_detector.py`  | Abstracción del modelo de pose + implementación MediaPipe + dibujo.  |
| `motion_tracker.py` | Cálculo de dirección/velocidad de movimiento entre frames.           |
| `requirements.txt`  | Dependencias.                                                        |

## Cómo se calcula la dirección de movimiento

1. En cada frame se toma un **punto de referencia** del cuerpo: el punto
   medio entre los hombros (con caderas y nariz como respaldo si los
   hombros no son visibles).
2. El punto se **suaviza con un promedio móvil exponencial (EMA)** para
   filtrar la vibración natural del modelo.
3. Se guarda un historial de ~0.4 s y la velocidad se calcula entre el
   punto más antiguo y el más reciente de esa ventana (robusto ante
   fluctuaciones de FPS).
4. Si la velocidad supera una **zona muerta** (~10 % del ancho del frame
   por segundo), se clasifica como izquierda/derecha y/o arriba/abajo;
   si no, la persona se considera quieta.

Los umbrales y la ventana se pueden ajustar en el constructor de
`MotionTracker` (`motion_tracker.py`).

## Cambiar el modelo de pose (p. ej. YOLOv8-pose)

La GUI no conoce MediaPipe: solo usa los contratos `BasePoseDetector` y
`PoseResult` definidos en `pose_detector.py`. Para usar otro modelo:

1. Crea una subclase de `BasePoseDetector` cuyo `process()` devuelva un
   `PoseResult` con un array `(N, 4)` de `[x, y, z, visibility]`
   normalizados.
2. Ajusta `SKELETON_CONNECTIONS` y `LIMB_GROUPS` al esquema de keypoints
   del nuevo modelo (p. ej. COCO-17 en YOLOv8-pose).
3. Cambia la clase instanciada en `VideoWorker.run()` (`gui.py`).

## Rendimiento

- La cámara se captura a 640x480, el mejor equilibrio precisión/FPS en CPU.
- Si el FPS es bajo, cambia `MODEL_COMPLEXITY = 0` en `gui.py`
  (modelo "lite": más rápido, algo menos preciso).

## Solución de problemas

| Problema | Solución |
| -------- | -------- |
| `pip` no encuentra `mediapipe` | Tu versión de Python no tiene rueda publicada; usa Python 3.11 o 3.12. |
| "No se pudo abrir la cámara" | Cierra otras apps que usen la cámara (Teams, Zoom, navegador) y pulsa **Buscar**. |
| `ImportError: DLL load failed` al cargar mediapipe | Instala el paquete "Microsoft Visual C++ Redistributable" más reciente. |
| El video va lento | Usa `MODEL_COMPLEXITY = 0` en `gui.py` y cierra aplicaciones pesadas. |
