# Ejecucion por consola

Guia rapida para generar imagenes y reportes desde consola. Esta guia sigue el criterio de los lineamientos INF-402: ejecucion reproducible, resultados visuales claros y demo practica para la presentacion final.

Ejecutar todos los comandos desde la raiz del repositorio:

```powershell
cd C:\Users\jgome\OneDrive\Documentos\GitHub\literatura
```

## 1. Preparar entorno

Instalar dependencias si el equipo aun no las tiene:

```powershell
python -m pip install ultralytics opencv-python pyyaml numpy torch
```

Verificar que existan la imagen de prueba y los modelos:

```powershell
Test-Path data\samples\sample-19.jpg
Test-Path models\tooth_piece_classifier\weights\best.pt
Test-Path models\treatment_detector\weights\best.pt
```

Estado actual esperado:

- `data\samples\sample-19.jpg`: debe ser `True`.
- `models\tooth_piece_classifier\weights\best.pt`: debe ser `True`.
- `models\treatment_detector\weights\best.pt`: sera `False` hasta copiar ahi el modelo de tratamiento entrenado.

## 2. Mostrar clasificacion de pieza dental

Usar este comando cuando quieras mostrar segmentacion/rotulado de piezas dentales con numero FDI.

```powershell
python src\predict_dental_xray.py `
  --model-role tooth `
  --image data\samples\sample-19.jpg `
  --output-dir results\demo_pieza_dental `
  --imgsz 1280 `
  --conf 0.25 `
  --device cpu
```

Salidas generadas:

```text
results\demo_pieza_dental\tooth_piece_classifier\sample-19_tooth_piece_classifier_labeled.jpg
results\demo_pieza_dental\tooth_piece_classifier\sample-19_tooth_piece_classifier_report.csv
```

Mostrar en la presentacion:

- La imagen `*_labeled.jpg` para la demostracion visual.
- El CSV `*_report.csv` como respaldo de predicciones, clases y confianza.

## 3. Mostrar posibles tratamientos o hallazgos

Usar este comando cuando quieras mostrar deteccion de hallazgos dentales asociados a posible tratamiento, por ejemplo caries, coronas, restauraciones, implantes o lesiones.

Requisito previo: debe existir este archivo:

```text
models\treatment_detector\weights\best.pt
```

Comando:

```powershell
python src\predict_dental_xray.py `
  --model-role treatment `
  --image data\samples\sample-19.jpg `
  --output-dir results\demo_tratamiento `
  --imgsz 1280 `
  --conf 0.25 `
  --device cpu
```

Salidas esperadas:

```text
results\demo_tratamiento\treatment_detector\sample-19_treatment_detector_labeled.jpg
results\demo_tratamiento\treatment_detector\sample-19_treatment_detector_report.csv
```

Si el comando falla con `No existe el modelo`, falta copiar el `.pt` entrenado en `models\treatment_detector\weights\best.pt`.

## 4. Mostrar ambos modelos en una misma demo

Usar esto para una demo completa: primero pieza dental y luego posible tratamiento/hallazgo sobre la misma radiografia.

Requisito: deben existir ambos modelos:

```powershell
Test-Path models\tooth_piece_classifier\weights\best.pt
Test-Path models\treatment_detector\weights\best.pt
```

Comando:

```powershell
python src\predict_dental_xray.py `
  --model-role both `
  --image data\samples\sample-19.jpg `
  --output-dir results\demo_integrada `
  --imgsz 1280 `
  --conf 0.25 `
  --device cpu
```

Salidas esperadas:

```text
results\demo_integrada\tooth_piece_classifier\sample-19_tooth_piece_classifier_labeled.jpg
results\demo_integrada\tooth_piece_classifier\sample-19_tooth_piece_classifier_report.csv
results\demo_integrada\treatment_detector\sample-19_treatment_detector_labeled.jpg
results\demo_integrada\treatment_detector\sample-19_treatment_detector_report.csv
results\demo_integrada\sample-19_combined_report.csv
```

## 5. Usar otra radiografia

Cambiar solo el valor de `--image`.

Ejemplo:

```powershell
python src\predict_dental_xray.py `
  --model-role tooth `
  --image data\samples\mi_radiografia.jpg `
  --output-dir results\demo_mi_radiografia `
  --imgsz 1280 `
  --conf 0.25 `
  --device cpu
```

La imagen debe existir antes de ejecutar el comando.

## 6. Generar auditoria de datos para materiales y metodos

Usar estos comandos si quieres mostrar que el flujo es reproducible y que los datasets fueron preparados localmente.

Auditoria/preparacion del dataset de pieza dental:

```powershell
$env:ANALYZE_ONLY="1"
$env:TRAIN_DATASET="teeth"
$env:PROJECT_DIR="results\data_preparation\tooth"
python src\dental_xray_pipeline.py
Remove-Item Env:\ANALYZE_ONLY
Remove-Item Env:\TRAIN_DATASET
Remove-Item Env:\PROJECT_DIR
```

Salidas principales:

```text
results\data_preparation\tooth\DATASET_ANALYSIS.md
results\data_preparation\tooth\dataset_audit.json
results\data_preparation\tooth\preparation_summary.json
```

Auditoria/preparacion del dataset de tratamientos/hallazgos:

```powershell
$env:ANALYZE_ONLY="1"
$env:TRAIN_DATASET="disease"
$env:PROJECT_DIR="results\data_preparation\treatment"
python src\dental_xray_pipeline.py
Remove-Item Env:\ANALYZE_ONLY
Remove-Item Env:\TRAIN_DATASET
Remove-Item Env:\PROJECT_DIR
```

Salidas principales:

```text
results\data_preparation\treatment\DATASET_ANALYSIS.md
results\data_preparation\treatment\dataset_audit.json
results\data_preparation\treatment\preparation_summary.json
results\data_preparation\treatment\prepared_yolo_seg\data.yaml
```

## 7. Configuracion de presentacion

Para la demo oral del proyecto final, conviene usar:

- `--imgsz 1280` para una imagen final mas presentable.
- `--conf 0.25` para evitar detecciones demasiado debiles en la demostracion.
- `--device cpu` si se quiere asegurar que funcione en este equipo sin depender de GPU.
- `--device 0` si se quiere usar GPU NVIDIA disponible.

La pauta exige mostrar resultados con figuras, tablas o metricas y una demostracion practica. Para eso, usar:

- Imagen rotulada `*_labeled.jpg` como figura principal.
- Reporte `*_report.csv` como tabla de predicciones.
- Archivos `DATASET_ANALYSIS.md` y `dataset_audit.json` como respaldo de materiales y metodos.
