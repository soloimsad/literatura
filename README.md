# Segmentacion de radiografias dentales - INF-402

Repositorio organizado segun las metricas y lineamientos del curso en
`docs/evaluation/00_metricas_evaluacion.md` y
`docs/evaluation/0-Lineamientos_PRIM_INF402.txt`.

## Estructura

- `src/`: codigo fuente principal en Python.
- `data/`: datasets locales, datos preparados e imagenes de prueba.
- `models/`: pesos y corridas de entrenamiento.
- `results/`: figuras, tablas, auditorias, predicciones y salidas generadas.
- `notebooks/`: flujo local para preparacion de datos, inferencia y demo.
- `docs/evaluation/`: pauta, rubrica y lineamientos del curso.
- `docs/literature_review/`: sintesis bibliografica y textos extraidos de articulos.
- `deliverables/`: carpetas de trabajo para revision bibliografica, paper IEEE TMI y presentacion final.
- `_revision_pendiente/`: material que no calza con la estructura evaluada y queda separado para revision.

## Instalacion

```powershell
python -m pip install ultralytics opencv-python pyyaml numpy torch
```

## Modelos locales

Los modelos se usan por rol, no por rutas sueltas:

- `tooth`: clasificacion/segmentacion de pieza dental.
  - Ruta: `models/tooth_piece_classifier/weights/best.pt`
  - Estado local: disponible.
- `treatment`: deteccion de posible tratamiento o hallazgo.
  - Ruta esperada: `models/treatment_detector/weights/best.pt`
  - Estado local: disponible.

La registry del proyecto esta en `src/model_registry.py` y
`models/model_registry.json`.

## Auditoria y preparacion de datos

```powershell
$env:ANALYZE_ONLY="1"
python src/dental_xray_pipeline.py
```

Por defecto el script busca datasets en `data/`, escribe resultados en `results/`
y prepara datos YOLO en `data/_prepared_teeth_yolo_seg`. Variables utiles:

- `TRAIN_DATASET=auto|teeth|disease`
- `LOCAL_DATA_DIR=<ruta>`
- `PROJECT_DIR=<ruta_resultados>`
- `EPOCHS=<numero>`
- `SOURCE_IMAGE=<ruta_imagen>`
- `BASE_MODEL=<ruta_en_models>` si quieres forzar otro peso local dentro de `models/`.

## Entrenamiento

### CNN propia para clasificacion de piezas dentales

Este es el flujo recomendado para el proyecto final si el objetivo es entrenar
un modelo deep learning con el dataset local de piezas dentales:

```powershell
python src/train_tooth_piece_cnn.py --prepare-only --force-prepare
```

Luego:

```powershell
python src/train_tooth_piece_cnn.py `
  --skip-prepare `
  --epochs 20 `
  --batch-size 64 `
  --image-size 128 `
  --device cpu
```

El codigo prepara recortes desde los poligonos del dataset y entrena una CNN en
PyTorch para clasificar piezas dentales. La metodologia esta documentada en
`docs/training_tooth_piece_cnn.md`.

### YOLO-seg local

```powershell
python src/dental_xray_pipeline.py
```

Las corridas de entrenamiento quedan en `models/tooth_piece_classifier/` o
`models/treatment_detector/` segun el dataset seleccionado. Los reportes o
predicciones generadas quedan en `results/`.

El entrenamiento tambien usa modelos locales: por defecto parte desde
`models/tooth_piece_classifier/weights/best.pt` para pieza dental o desde
`models/treatment_detector/weights/best.pt` para tratamientos/hallazgos.

## Inferencia

```powershell
python src/predict_dental_xray.py `
  --model-role tooth `
  --image data/samples/sample-19.jpg `
  --output-dir results/inference
```

Para ejecutar el modelo de tratamiento cuando el peso exista:

```powershell
python src/predict_dental_xray.py `
  --model-role treatment `
  --image data/samples/sample-19.jpg `
  --output-dir results/inference
```

Para correr ambos modelos:

```powershell
python src/predict_dental_xray.py `
  --model-role both `
  --image data/samples/sample-19.jpg `
  --output-dir results/inference
```

Cada comando genera una imagen rotulada y un reporte CSV.

## Metricas desde checkpoints

Para extraer las metricas guardadas dentro de los pesos locales:

```powershell
python src/extract_checkpoint_metrics.py
```

Esto genera:

- `results/metrics/checkpoint_metrics.json`
- `results/metrics/checkpoint_metrics.md`

El script calcula F1 desde precision y recall. DSC/Dice no viene guardado
directamente en los checkpoints; debe recalcularse con predicciones y mascaras
ground truth compatibles.

## Notebooks

Orden sugerido para preparar la presentacion:

1. `notebooks/01_preparacion_datos_local.ipynb`
2. `notebooks/02_inferencia_pieza_dental.ipynb`
3. `notebooks/03_inferencia_tratamiento.ipynb`
4. `notebooks/04_demo_integrada_local.ipynb`

Todo el flujo esta orientado a ejecucion local en este equipo.

## Entregables del curso

- Revision bibliografica: `deliverables/revision_bibliografica/`
- Manuscrito IEEE TMI: `deliverables/manuscrito_ieee_tmi/`
- Presentacion final: `deliverables/presentacion_final/`

Los datos pesados, modelos y resultados generados se mantienen locales mediante
`.gitignore`. No se eliminaron archivos del proyecto; los elementos fuera de la
estructura exigida fueron separados en `_revision_pendiente/`.
