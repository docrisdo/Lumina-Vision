# Lumina Vision

Proyecto listo para GitHub y para clonar en Raspberry Pi 4 con:

- camara IMX708 Camera Module 3 via `Picamera2`
- deteccion de objetos con TensorFlow Lite
- OCR con Tesseract en espanol
- texto a voz offline en espanol

El modulo de ultrasonicos queda fuera por ahora.

## Objetivo tecnico

Esta base prioriza viabilidad real en Raspberry Pi 4 de 4 GB:

- captura moderada para no saturar CPU
- detector liviano en TFLite
- OCR por intervalos, no por frame
- voz offline sin depender de internet

## Estructura

```text
Lumina-Vision/
|- main.py
|- .env.example
|- .gitignore
|- requirements.txt
|- requirements-dev.txt
|- pyproject.toml
|- deploy/
|  |- lumina-vision.service
|- models/
|  |- coco_labels.txt
|  |- README.md
|- scripts/
|  |- bootstrap_pi.sh
|  |- run_pi.sh
|  |- validate_runtime.py
|  |- download_lumina_assets.py
|- src/
|  |- lumina_vision/
|     |- app.py
|     |- camera.py
|     |- config.py
|     |- ocr.py
|     |- pipeline.py
|     |- speech.py
|     |- utils.py
|     |- detectors/
|        |- tflite_detector.py
|- tests/
   |- test_config.py
```

## Flujo recomendado

### 1. Subir a GitHub desde Windows

Dentro de `C:\Users\allan\Proyecto_Karla\Lumina-Vision`:

```powershell
git init
git branch -M main
git add .
git commit -m "Proyecto Lumina Vision listo para Raspberry Pi"
git remote add origin https://github.com/TU_USUARIO/Lumina-Vision.git
git push -u origin main
```

### 2. Clonar en la Raspberry

```bash
cd ~
git clone https://github.com/TU_USUARIO/Lumina-Vision.git
cd Lumina-Vision
bash scripts/bootstrap_pi.sh
```

### 3. Colocar el modelo de deteccion

Pon tu modelo en:

```text
models/efficientdet_lite0.tflite
```

Si usas otro nombre o ruta, cambialo en `.env`.

### 4. Configurar variables

```bash
cp .env.example .env
nano .env
```

Para Raspberry sin monitor, deja:

```env
LUMINA_SHOW_PREVIEW=false
```

### 5. Ejecutar

```bash
bash scripts/run_pi.sh
```

## Dependencias del sistema

El script `scripts/bootstrap_pi.sh` instala:

- `python3-picamera2`
- `python3-libcamera`
- `python3-venv`
- `tesseract-ocr`
- `tesseract-ocr-spa`
- `espeak-ng`
- `libespeak1`
- `ffmpeg`

## Configuracion importante

Variables clave del `.env`:

- `LUMINA_ENABLE_OBJECT_DETECTION=true`
- `LUMINA_ENABLE_OCR=true`
- `LUMINA_ENABLE_TTS=true`
- `LUMINA_OCR_LANGUAGE=spa+eng`
- `LUMINA_SHOW_PREVIEW=false` para modo headless
- `LUMINA_DETECTION_RUN_EVERY_N_FRAMES=2`
- `LUMINA_OCR_RUN_INTERVAL_SECONDS=2.0`

## Servicio automatico al encender

Si luego quieres que arranque solo:

```bash
sudo cp deploy/lumina-vision.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lumina-vision
sudo systemctl start lumina-vision
```

Antes revisa que la ruta del usuario `pi` coincida con tu Raspberry.

## Recomendaciones para tu hardware

- Usa `1280x720` para el pipeline principal, no 12 MP.
- No corras OCR en todos los frames.
- Si la voz habla demasiado, aumenta `LUMINA_SPEECH_COOLDOWN_SECONDS`.
- Si no usaras pantalla, evita `cv2.imshow` con `LUMINA_SHOW_PREVIEW=false`.

## Desarrollo en Windows

En Windows el sistema hace fallback a `cv2.VideoCapture` porque `Picamera2` no esta disponible.
Eso te permite desarrollar la estructura del proyecto en VS Code, pero la prueba real de camara debe hacerse en la Raspberry.
