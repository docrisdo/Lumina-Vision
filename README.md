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

Para probar una hoja o libro:

```bash
source .venv/bin/activate
python scripts/test_ocr_capture.py
```

La ventana muestra una guia de enfoque. Usa este flujo:

- Coloca la hoja dentro del rectangulo amarillo.
- Presiona `F` para enfocar.
- Espera a que indique enfoque aceptable o bueno.
- Presiona `ESPACIO` para capturar.

Archivos de diagnostico:

- `debug_ocr/ocr_original.jpg`: captura cruda.
- `debug_ocr/ocr_best_for_tesseract.jpg`: variante para texto grande.
- `debug_ocr/ocr_page_variant_*.jpg`: variantes para lectura de pagina.

Si el texto se ve borroso en la ventana, el OCR no va a leer bien. Primero calibra enfoque.

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

Si aparece `echo pin set high`, revisa `ECHO`, divisor de voltaje, GND comun y que `TRIG/ECHO` no esten invertidos.

## Voz

Piper es la opcion recomendada por claridad:

```bash
source .venv/bin/activate
pip install piper-tts
bash scripts/download_piper_spanish_voice.sh
python scripts/test_audio.py
```

La app usa cache de frases para que avisos repetidos salgan mas rapido. Las alertas del ultrasonico tienen prioridad sobre OCR/objetos.

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
