# Lumina Vision

Lentes inteligentes de asistencia para personas con discapacidad visual. El prototipo corre localmente en Raspberry Pi 4 con camara IMX708 autofocus, deteccion de objetos, OCR para hojas/libros y voz en espanol por audifonos.

El modulo de sensores ultrasonicos queda fuera por ahora.

## Modo Principal: Lentes

El sistema esta configurado para funcionar sin teclado:

- inicia camara, OCR, deteccion y voz automaticamente
- prioriza leer texto cercano de hojas o libros
- anuncia objetos escolares comunes sin repetirlos de forma molesta
- usa voz offline en espanol, preferentemente con Piper
- evita `cv2.imshow` por defecto para no cargar la Raspberry

Para pruebas con pantalla puedes cambiar en `.env`:

```env
LUMINA_SHOW_PREVIEW=true
LUMINA_WEARABLE_MODE=false
```

## Hardware Objetivo

- Raspberry Pi 4 Modelo B, 4 GB RAM
- Arducam 12MP IMX708 Autofocus Camera Module 3 NoIR
- Audifonos o bocina conectados a la Raspberry
- Modelo TFLite de deteccion en `models/efficientdet_lite0.tflite`
- Sensores HC-SR04 pendientes para una fase posterior

## Instalacion En Raspberry

```bash
cd ~
git clone https://github.com/TU_USUARIO/Lumina-Vision.git
cd Lumina-Vision
bash scripts/bootstrap_pi.sh
```

Descarga los modelos:

```bash
bash scripts/download_lumina_assets.py
bash scripts/download_piper_spanish_voice.sh
```

Si `download_lumina_assets.py` no se ejecuta directo, usa:

```bash
python scripts/download_lumina_assets.py
```

Configura variables:

```bash
cp .env.example .env
nano .env
```

Ejecuta:

```bash
bash scripts/run_pi.sh
```

## Configuracion Recomendada Para Lentes

```env
LUMINA_WEARABLE_MODE=true
LUMINA_SHOW_PREVIEW=false
LUMINA_ENABLE_OCR=true
LUMINA_ENABLE_OBJECT_DETECTION=true
LUMINA_ENABLE_TTS=true
LUMINA_TTS_ENGINE=piper
LUMINA_OCR_AUTO_READ=true
LUMINA_OCR_STABLE_READS=2
LUMINA_CAMERA_REFOCUS_BEFORE_OCR=true
```

Para diagnostico visual:

```env
LUMINA_WEARABLE_MODE=false
LUMINA_SHOW_PREVIEW=true
```

## OCR

El OCR esta optimizado para texto cercano, como hojas, libros y etiquetas escolares. La camara debe apuntar al texto a una distancia aproximada de 20 a 50 cm. El sistema reenfoca de forma limitada para evitar trabas y lee solo cuando el texto se mantiene estable.

Si no lee texto:

- mejora la iluminacion
- acerca o aleja la hoja lentamente
- evita movimiento mientras enfoca
- prueba con texto grande y alto contraste
- baja `LUMINA_OCR_MIN_SHARPNESS` si rechaza demasiado
- sube `LUMINA_OCR_STABLE_READS` si lee basura

## Voz

Piper es la opcion recomendada porque suena mas natural que `espeak-ng`.

```bash
source .venv/bin/activate
pip install piper-tts
bash scripts/download_piper_spanish_voice.sh
python scripts/test_audio.py
```

La app usa cache para frases repetidas, por eso frases como "Veo un libro" salen mas rapido despues de la primera vez.

## Objetos Escolares

El detector prioriza objetos utiles en escuela cuando el modelo los reconoce:

- persona
- libro
- mochila
- laptop
- celular
- botella
- tijeras
- teclado
- raton
- silla
- mesa
- reloj

Limitacion importante: el modelo COCO no reconoce todos los utiles escolares, por ejemplo lapiz, cuaderno o regla. Para eso se necesita entrenar o agregar un modelo personalizado en una fase posterior.

## Pruebas Basicas

Compilar:

```bash
python -m compileall src scripts
```

Probar camara:

```bash
rpicam-hello -t 5000
```

Probar voz:

```bash
source .venv/bin/activate
python scripts/test_audio.py
```

Probar OCR con una captura guardada:

```bash
source .venv/bin/activate
python scripts/test_ocr_capture.py
```

La imagen usada por OCR queda en `debug_ocr/ocr_original.jpg`. Si no lee, abre esa imagen y revisa si el texto realmente sale enfocado, grande y con buena luz.

Probar programa:

```bash
bash scripts/run_pi.sh
```

## Servicio Automatico

Edita `deploy/lumina-vision.service` si tu usuario no es `pi`. Luego:

```bash
sudo cp deploy/lumina-vision.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lumina-vision
sudo systemctl start lumina-vision
```

## Notas De Rendimiento

- No uses 12 MP para el pipeline; `1280x720` es mas viable en Raspberry Pi 4.
- OCR es pesado: corre por intervalos y en hilo separado.
- Si se siente lento, sube `LUMINA_OCR_RUN_INTERVAL_SECONDS` a `3.5`.
- Si la voz se acumula, baja `LUMINA_SPEECH_MAX_QUEUE_SIZE` a `1`.
- Si la camara da errores `Camera frontend has timed out`, revisa flex CSI, alimentacion y que no haya otra app usando la camara.
