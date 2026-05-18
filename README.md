# Lumina Vision

Lentes inteligentes de asistencia para personas con discapacidad visual. El prototipo corre localmente en Raspberry Pi 4 con camara IMX708 autofocus, OCR para leer hojas/libros, voz en espanol por audifonos y alerta de proximidad con sensor ultrasonico HC-SR04.

La prioridad actual del proyecto es lectura de texto escolar. La deteccion de objetos queda como apoyo, no como funcion principal.

## Funciones

- Lectura de texto en espanol usando OCR con Tesseract.
- Voz offline en espanol, recomendada con Piper.
- Camara IMX708 autofocus con calibracion manual de enfoque.
- Modo lectura de pagina para hojas/libros escolares.
- Alerta de proximidad por audio con HC-SR04.
- Deteccion de objetos escolares comunes cuando el modelo TFLite los reconoce.
- Modo lentes sin teclado ni ventana de preview.

## Hardware

- Raspberry Pi 4 Modelo B, 4 GB RAM.
- Arducam 12MP IMX708 Autofocus Camera Module 3 NoIR.
- Audifonos o bocina conectados a la Raspberry.
- Sensor ultrasonico HC-SR04.
- Divisor de voltaje para `ECHO` del HC-SR04.
- Modelo TFLite en `models/efficientdet_lite0.tflite` si se usa deteccion de objetos.

Importante: el pin `ECHO` del HC-SR04 entrega 5V. No debe conectarse directo a la Raspberry. Usa divisor de voltaje o level shifter para bajarlo a 3.3V.

## Instalacion En Raspberry

```bash
cd ~
git clone https://github.com/docrisdo/Lumina-Vision.git
cd Lumina-Vision
bash scripts/bootstrap_pi.sh
```

Descarga modelos:

```bash
python scripts/download_lumina_assets.py
bash scripts/download_piper_spanish_voice.sh
```

Configura variables:

```bash
cp .env.example .env
nano .env
```

## Librerias Utilizadas

Dependencias principales de Python:

- `numpy`: manejo de arreglos e imagenes para OCR y deteccion.
- `opencv-python` (`cv2`): captura/preview, procesamiento de imagenes, recortes, nitidez, binarizacion y anotaciones visuales.
- `pillow`: soporte general para imagenes y compatibilidad con utilidades de vision.
- `python-dotenv`: carga de configuracion desde `.env`.
- `loguru`: logs claros del sistema, camara, OCR, voz y sensores.
- `pytesseract`: conexion entre Python y Tesseract OCR para extraer texto de hojas/libros.
- `pyttsx3`: motor TTS alternativo si Piper o `espeak-ng` no estan disponibles.
- `requests`: descarga de modelos y assets desde scripts de instalacion.
- `gpiozero`: lectura del sensor ultrasonico HC-SR04 en Raspberry Pi.

Dependencias del sistema en Raspberry:

- `python3-picamera2` y `python3-libcamera`: control de la camara IMX708 y autofocus.
- `tesseract-ocr` y `tesseract-ocr-spa`: motor OCR y paquete de idioma espanol.
- `piper-tts`: voz local mas natural en espanol.
- `espeak-ng`: voz alternativa y respaldo si Piper no esta disponible.
- `pulseaudio-utils`, `aplay`/`paplay`: reproduccion de audio por audifonos o bocina.
- `ffmpeg`: soporte auxiliar de audio/multimedia.
- `tflite_runtime`, `ai-edge-litert` o `tensorflow.lite`: runtime para modelos TFLite de deteccion de objetos.

Librerias estandar de Python usadas internamente:

- `threading` y `queue`: OCR, voz y ultrasonico en hilos separados para no bloquear el programa.
- `subprocess` y `shutil`: ejecucion de Piper, `espeak-ng`, `aplay`/`paplay` y validacion de comandos.
- `time`: cooldowns, intervalos y mediciones.
- `pathlib`, `os`, `sys`: manejo de rutas, entorno y arranque de scripts.
- `dataclasses`: estructuras de configuracion y resultados.
- `re`: limpieza y validacion de texto OCR.
- `hashlib`: cache de audios generados por Piper.

## Ejecucion Recomendada

Modo lectura con OCR, voz y ultrasonico:

```bash
bash scripts/run_reading.sh
```

Modo general:

```bash
bash scripts/run_pi.sh
```

Preview para pruebas:

```bash
bash scripts/run_preview.sh
```

## OCR Y Lectura De Paginas

Primero calibra OCR con un texto grande y conocido:

```bash
source .venv/bin/activate
python scripts/calibrate_ocr.py --expected HOLA
```

Cuando eso funcione, calibra con una frase de una hoja real:

```bash
python scripts/calibrate_ocr.py --expected "El cuervo y la jarra"
python scripts/calibrate_ocr.py --expected "Un cuervo sediento encontro una jarra"
```

El script prueba varias posiciones de lente y al final muestra un valor como:

```env
LUMINA_CAMERA_LENS_POSITION=4.5
```

Pon ese valor en `.env`. Para lectura, una hoja/libro debe estar estable, bien iluminado y llenar la mayor parte del rectangulo de guia. Si el texto se ve borroso en `debug_ocr_calibration/best_ocr_original.jpg`, el OCR no va a leer bien aunque Tesseract este instalado.

Si el modulo quedo girado en los lentes, corrige la orientacion por software:

```env
LUMINA_CAMERA_ROTATION=0
```

Valores validos: `0`, `90`, `180`, `270`.

El OCR solo procesa el rectangulo amarillo para evitar basura del fondo:

```env
LUMINA_OCR_ROI_X1=0.22
LUMINA_OCR_ROI_Y1=0.06
LUMINA_OCR_ROI_X2=0.78
LUMINA_OCR_ROI_Y2=0.94
LUMINA_OCR_FAST_MODE=true
```

Para probar una hoja o libro:

```bash
source .venv/bin/activate
python scripts/test_ocr_capture.py
```

La ventana muestra una guia de enfoque. Usa este flujo:

- Si la camara se ve girada, presiona `R` hasta verla derecha. Copia ese valor a `LUMINA_CAMERA_ROTATION` en `.env`.
- El contorno verde indica que la hoja fue detectada automaticamente y se procesara aunque no este centrada.
- El rectangulo amarillo es solo guia/fallback si no se detecta la hoja.
- Presiona `F` para enfocar.
- Espera a que indique enfoque aceptable o bueno.
- Presiona `ESPACIO` para capturar.

Archivos de diagnostico:

- `debug_ocr/ocr_original.jpg`: captura cruda.
- `debug_ocr/ocr_best_for_tesseract.jpg`: variante para texto grande.
- `debug_ocr/ocr_word_boxes_best.jpg`: palabras que Tesseract acepto con confianza alta. Verde significa palabra util; rojo significa posible ruido.
- `debug_ocr/ocr_region_*.jpg`: regiones usadas por el OCR, incluyendo recorte automatico de documento si se detecta.
- `debug_ocr/ocr_*_page_variant_*.jpg`: variantes para lectura de pagina.

Si `ocr_best_for_tesseract.jpg` se ve claro pero `ocr_word_boxes_best.jpg` casi no marca palabras, el problema es de preprocesamiento/Tesseract. Si las cajas aparecen sobre fondo, manos o bordes, el problema es de deteccion de region. Si el texto se ve borroso en la ventana, primero calibra enfoque.

## Calibrar Enfoque

Para lectura de hojas, conviene fijar una posicion de lente en vez de depender siempre del autofocus:

```bash
source .venv/bin/activate
python scripts/calibrate_focus.py
```

El script devuelve algo como:

```text
LUMINA_CAMERA_LENS_POSITION=4.0
```

Agrega ese valor en `.env`:

```env
LUMINA_CAMERA_LENS_POSITION=4.0
```

Referencia practica: `4.0` suele estar cerca de 25 cm y `2.5` cerca de 40 cm, pero el valor correcto depende del montaje.

## Sensor Ultrasonico

Cableado por defecto:

```text
VCC  -> 5V
GND  -> GND
TRIG -> GPIO23
ECHO -> GPIO24 con divisor de voltaje a 3.3V
```

Divisor recomendado:

```text
ECHO sensor -> resistencia 1k -> GPIO24
GPIO24 -> resistencia 2k -> GND
```

Prueba solo distancia:

```bash
source .venv/bin/activate
python scripts/test_ultrasonic.py
```

Prueba distancia con audio:

```bash
source .venv/bin/activate
python scripts/test_ultrasonic_audio.py
```

Por defecto avisa cuando un objeto esta a menos de 15 cm:

```env
LUMINA_ULTRASONIC_ALERT_DISTANCE_CM=15
```

Mensaje de alerta:

```text
Cuidado. Hay un objeto a 12 centimetros.
```

La alerta usa una voz rapida tipo lector de pantalla para usuarios con discapacidad visual:

```env
LUMINA_ULTRASONIC_SPEECH_RATE=500
```

Si se entiende bien y quieres mas velocidad, prueba `600`. Si es demasiado rapida, baja a `420`.

Si aparece `echo pin set high`, revisa `ECHO`, divisor de voltaje, GND comun y que `TRIG/ECHO` no esten invertidos.

## Voz

Piper es la opcion recomendada por claridad:

```bash
source .venv/bin/activate
pip install piper-tts
bash scripts/download_piper_spanish_voice.sh
python scripts/test_audio.py
```

La app usa cache de frases para que avisos repetidos salgan mas rapido. Las alertas del ultrasonico tienen prioridad sobre OCR/objetos y usan una ruta rapida con `espeak-ng` para evitar esperas de Piper.

## Objetos Escolares

El detector puede anunciar objetos comunes si el modelo los reconoce:

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

Limitacion: con COCO no se reconocen todos los utiles escolares, como lapiz, cuaderno o regla. Para eso se necesita un modelo personalizado.

## Variables Principales

```env
LUMINA_WEARABLE_MODE=true
LUMINA_SHOW_PREVIEW=false
LUMINA_ENABLE_OCR=true
LUMINA_ENABLE_TTS=true
LUMINA_ENABLE_ULTRASONIC=true
LUMINA_ENABLE_OBJECT_DETECTION=false
LUMINA_TTS_ENGINE=piper
LUMINA_OCR_PAGE_MODE=true
LUMINA_ULTRASONIC_ALERT_DISTANCE_CM=15
```

Para pruebas con ventana:

```env
LUMINA_WEARABLE_MODE=false
LUMINA_SHOW_PREVIEW=true
```

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

Probar OCR:

```bash
source .venv/bin/activate
python scripts/test_ocr_capture.py
```

Probar ultrasonico:

```bash
source .venv/bin/activate
python scripts/test_ultrasonic_audio.py
```

## Notas De Rendimiento

- Para OCR de pagina se usa mas resolucion, pero menos FPS.
- OCR corre en hilo separado para no bloquear la camara.
- El ultrasonico corre en hilo separado y no debe bloquear OCR.
- La alerta ultrasonica limpia la cola de voz para responder mas rapido.
- Si la Raspberry se siente lenta, deja objetos desactivados hasta que OCR este estable.
- Si la camara da `Camera frontend has timed out`, revisa flex CSI, alimentacion y que no haya otra app usando la camara.

## Servicio Automatico

Edita `deploy/lumina-vision.service` si tu usuario no es `pi`. Luego:

```bash
sudo cp deploy/lumina-vision.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lumina-vision
sudo systemctl start lumina-vision
```
