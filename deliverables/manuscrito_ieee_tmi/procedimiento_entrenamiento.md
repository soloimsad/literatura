# Procedimiento de entrenamiento de los modelos

Este documento explica el procedimiento usado para preparar los datos y entrenar los modelos del proyecto. Esta escrito con foco en la pauta de INF-402: metodologia replicable, arquitectura justificada, metricas coherentes con la tarea y uso local del repositorio.

## Resumen general

El proyecto utiliza dos modelos entrenados con YOLO-seg mediante la libreria Ultralytics:

| Modelo | Rol dentro del sistema | Ruta local del peso final |
|---|---|---|
| Pieza dental | Detectar/segmentar piezas dentales y asignar la clase de pieza correspondiente | `models/tooth_piece_classifier/weights/best.pt` |
| Posible tratamiento o hallazgo | Detectar regiones asociadas a caries, coronas, restauraciones, implantes, lesiones u otros hallazgos dentales | `models/treatment_detector/weights/best.pt` |

Ambos modelos se ejecutan de forma local en este equipo. El proyecto ya no depende de Google Drive ni de rutas externas para la inferencia o el entrenamiento reproducible. Las rutas oficiales estan declaradas en `src/model_registry.py` y resumidas en `models/model_registry.json`.

## Tipo de red utilizada

Los modelos corresponden a YOLO con tarea de segmentacion, es decir, modelos de deteccion de objetos capaces de predecir:

- clase del objeto;
- confianza de la prediccion;
- caja delimitadora;
- mascara o poligono de segmentacion cuando el peso fue entrenado para segmentacion.

YOLO no es una CNN clasica de clasificacion global, porque no entrega una sola etiqueta para toda la imagen. Es una arquitectura moderna de deteccion/segmentacion basada en redes convolucionales, pensada para localizar multiples objetos dentro de una misma imagen. En este proyecto eso es importante porque una radiografia panoramica contiene muchas piezas dentales y varios posibles hallazgos en una sola imagen.


## Datasets usados

El entrenamiento se separa en dos fuentes de datos porque las tareas no son iguales.

### Dataset para pieza dental

El modelo de pieza dental usa el dataset local:

```text
data/teeth_segmentation/
```

Este dataset contiene radiografias dentales con anotaciones en formato Supervisely. Las anotaciones vienen como archivos JSON con poligonos por pieza dental. El pipeline convierte estas anotaciones a formato YOLO-seg para poder entrenar con Ultralytics.

La auditoria local generada por el proyecto indica:

| Split | Imagenes | Labels | Objetos/poligonos |
|---|---:|---:|---:|
| Train | 478 | 478 | 12198 |
| Valid | 59 | 59 | 1518 |
| Test | 61 | 61 | 1602 |

Total auditado:

- 598 imagenes;
- 598 anotaciones;
- 15318 objetos o poligonos;
- 32 clases numericas.


### Dataset para posible tratamiento o hallazgo

El modelo de tratamiento/hallazgo usa el dataset local:

```text
data/dental-disease-panoramic-detection-dataset/
```

Este dataset ya viene en formato YOLO, dentro de:

```text
data/dental-disease-panoramic-detection-dataset/YOLO/YOLO/
```

La estructura contiene carpetas `train`, `valid` y `test`, cada una con `images/` y `labels/`.

La revision local de archivos muestra:

| Split | Imagenes | Labels |
|---|---:|---:|
| Train | 9481 | 9481 |
| Valid | 2871 | 2871 |
| Test | 1580 | 1580 |

Este dataset contiene 31 clases de hallazgos o condiciones dentales, entre ellas:

- Caries;
- Crown;
- Filling;
- Implant;
- Missing teeth;
- Periapical lesion;
- Root Canal Treatment;
- Bone Loss;
- Fracture teeth;
- Primary teeth.

## Entrenamiento

El entrenamiento se realiza con Ultralytics YOLO desde `src/dental_xray_pipeline.py`. El flujo general es:

1. Se audita el dataset.
2. Se genera o valida el archivo `data.yaml`.
3. Se selecciona el dispositivo de entrenamiento:
   - GPU si `torch.cuda.is_available()` devuelve verdadero;
   - CPU en caso contrario.
4. Se carga el modelo base local desde `models/`.
5. Se ejecuta `model.train(...)`.
6. Se guarda la corrida dentro de `models/`.
7. Se conserva el mejor peso en `weights/best.pt`.
8. Se generan predicciones de demostracion y reportes dentro de `results/`.

Los hiperparametros por defecto definidos en el script son:

| Parametro | Valor por defecto | Descripcion |
|---|---:|---|
| `EPOCHS` | 15 | Numero de epocas de entrenamiento |
| `IMGSZ` | 512 | Tamano de imagen usado durante entrenamiento |
| `BATCH` | 16 | Tamano de batch |
| `WORKERS` | 2 | Procesos de carga de datos |
| `PATIENCE` | 6 | Paciencia para early stopping |
| `SEED` | 42 | Semilla para reproducibilidad |
| `AMP` | Automatico | Se activa si hay CUDA disponible |

El entrenamiento guarda los resultados en:

```text
models/tooth_piece_classifier/
```

o:

```text
models/treatment_detector/
```

dependiendo del valor de `TRAIN_DATASET` y `RUN_NAME`.



### Entrenar modelo de tratamiento/hallazgo

```powershell
$env:TRAIN_DATASET="disease"
$env:RUN_NAME="treatment_detector"
$env:EPOCHS="15"
$env:IMGSZ="512"
$env:BATCH="16"
python src/dental_xray_pipeline.py
```

Peso final esperado:

```text
models/treatment_detector/weights/best.pt
```


