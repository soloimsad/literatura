# Segmentacion de radiografias dentales - INF-402

Repositorio organizado segun las metricas y lineamientos del curso en
`docs/evaluation/00_metricas_evaluacion.md` y
`docs/evaluation/0-Lineamientos_PRIM_INF402.txt`.

## Estructura

- `src/`: codigo fuente principal en Python.
- `data/`: datasets locales, datos preparados e imagenes de prueba.
- `models/`: pesos y corridas de entrenamiento.
- `results/`: figuras, tablas, auditorias, predicciones y salidas generadas.
- `docs/evaluation/`: pauta, rubrica y lineamientos del curso.
- `docs/literature_review/`: sintesis bibliografica y textos extraidos de articulos.
- `deliverables/`: carpetas de trabajo para revision bibliografica, paper IEEE TMI y presentacion final.
- `_revision_pendiente/`: material que no calza con la estructura evaluada y queda separado para revision.

## Instalacion

```powershell
python -m pip install ultralytics opencv-python pyyaml numpy torch
```

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

## Entrenamiento

```powershell
python src/dental_xray_pipeline.py
```

Las corridas de entrenamiento quedan en `models/training_runs/` y los reportes
o predicciones generadas quedan en `results/`.

## Inferencia

```powershell
python src/predict_dental_xray.py `
  --model models/training_runs/dental_xray_cnn_segmentation/weights/best.pt `
  --image data/samples/test.jpg `
  --output-dir results/inference
```

El comando genera una imagen segmentada/rotulada y un reporte CSV con las
detecciones.

## Entregables del curso

- Revision bibliografica: `deliverables/revision_bibliografica/`
- Manuscrito IEEE TMI: `deliverables/manuscrito_ieee_tmi/`
- Presentacion final: `deliverables/presentacion_final/`

Los datos pesados, modelos y resultados generados se mantienen locales mediante
`.gitignore`. No se eliminaron archivos del proyecto; los elementos fuera de la
estructura exigida fueron separados en `_revision_pendiente/`.
