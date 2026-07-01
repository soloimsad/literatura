# Roadmap del proyecto: clasificacion de piezas dentales

Este roadmap describe que se hace en el proyecto, en que orden se ejecuta y que resultados se obtienen en cada etapa. El foco actual del proyecto final es entrenar un modelo deep learning propio para **clasificar piezas dentales** usando el dataset local disponible.

## Objetivo final

Entrenar una CNN supervisada que reciba un recorte de una pieza dental y prediga la clase correspondiente. El dataset original trae radiografias completas con anotaciones por poligono; por eso el primer paso es transformar esas anotaciones en muestras individuales de entrenamiento.

## Roadmap general

| Etapa | Que se hace | Codigo/archivo | Resultado |
|---|---|---|---|
| 1. Revision del dataset | Se revisa el dataset de radiografias dentales en formato Supervisely. | `data/teeth_segmentation/` | Se confirma que existen 598 radiografias con anotaciones JSON. |
| 2. Extraccion por poligonos | Cada poligono anotado se usa para aislar una pieza dental. | `src/train_tooth_piece_cnn.py` | Se generan recortes individuales por diente. |
| 3. Preparacion de splits | Las radiografias se dividen en entrenamiento, validacion y prueba. | `src/train_tooth_piece_cnn.py` | Se evita fuga de informacion porque el split se hace por radiografia, no por recorte. |
| 4. Entrenamiento CNN | Se entrena una CNN propia en PyTorch sobre los recortes. | `src/train_tooth_piece_cnn.py` | Se genera un modelo `best.pt` segun macro F1 de validacion. |
| 5. Evaluacion | Se calculan metricas de clasificacion. | `metrics.json`, `training_history.csv`, matrices de confusion | Se obtiene accuracy, macro precision, macro recall y macro F1. |
| 6. Presentacion de resultados | Se reporta la metodologia, el dataset preparado y las metricas finales. | `docs/training_tooth_piece_cnn.md` y salidas del modelo | Se arma la seccion de resultados para presentacion/manuscrito. |

## Etapa 1: dataset usado

El dataset usado para el modelo de piezas dentales esta en:

```text
data/teeth_segmentation/
```

La fuente principal es:

```text
data/teeth_segmentation/Teeth Segmentation JSON/d2/
```

Este directorio contiene:

- `img/`: radiografias dentales.
- `ann/`: anotaciones JSON.
- `masks_human/`: mascaras humanas disponibles en el dataset.
- `masks_machine/`: mascaras generadas automaticamente.

El flujo actual usa las anotaciones `ann/`, porque cada JSON contiene objetos tipo poligono con una clase asociada.

## Etapa 2: por que se usan poligonos

Los poligonos son importantes porque el objetivo no es clasificar la radiografia completa, sino una pieza dental especifica. Una radiografia panoramica contiene multiples dientes, por lo que una unica etiqueta por imagen no sirve para entrenar un clasificador de piezas.

El poligono permite:

- identificar donde esta cada diente;
- recortar cada pieza como una muestra individual;
- eliminar parte del fondo con una mascara;
- asociar cada recorte a una clase clara;
- entrenar una CNN con datos supervisados.

En terminos metodologicos, esto se conecta con los papers de segmentacion e identificacion dental: primero se separa la estructura dental y luego se aprende su identidad.

## Etapa 3: preparacion de datos

El comando de preparacion es:

```powershell
python src/train_tooth_piece_cnn.py --prepare-only --force-prepare
```

Este comando genera:

```text
data/_prepared_tooth_piece_classifier/
```

Dentro quedan:

- `train/`: recortes de entrenamiento organizados por clase.
- `valid/`: recortes de validacion organizados por clase.
- `test/`: recortes de prueba organizados por clase.
- `manifest.csv`: inventario completo de recortes, clase, imagen fuente y anotacion fuente.
- `preparation_summary.json`: resumen tecnico de la preparacion.

## Resultados actuales de preparacion

La preparacion completa ya fue ejecutada localmente. Los resultados actuales son:

| Split | Radiografias | Recortes dentales |
|---|---:|---:|
| Train | 478 | 12198 |
| Valid | 59 | 1518 |
| Test | 61 | 1602 |
| Total | 598 | 15318 |

Tambien se detectaron 32 clases:

```text
1, 2, 3, ..., 32
```

No se registraron objetos omitidos en la preparacion (`skipped: {}`), por lo que las anotaciones validas fueron convertidas correctamente.

## Etapa 4: entrenamiento del modelo

El entrenamiento completo se ejecuta con:

```powershell
python src/train_tooth_piece_cnn.py `
  --skip-prepare `
  --epochs 20 `
  --batch-size 64 `
  --image-size 128 `
  --device cpu
```

Si existe GPU compatible:

```powershell
python src/train_tooth_piece_cnn.py `
  --skip-prepare `
  --epochs 20 `
  --batch-size 64 `
  --image-size 128 `
  --device cuda
```

El modelo implementado se llama `ToothPieceCNN`. Es una CNN propia construida en PyTorch, con:

- entrada en escala de grises;
- bloques convolucionales;
- BatchNorm;
- ReLU;
- MaxPooling;
- Adaptive Average Pooling;
- capas fully connected;
- CrossEntropyLoss para clasificacion multiclase.

## Etapa 5: resultados que genera el entrenamiento

Cuando se ejecuta el entrenamiento completo, se generan archivos en:

```text
models/tooth_piece_cnn_classifier/
```

Resultados esperados:

| Archivo | Funcion |
|---|---|
| `best.pt` | Mejor modelo segun macro F1 de validacion. |
| `last.pt` | Ultimo checkpoint entrenado. |
| `metrics.json` | Metricas finales del conjunto test. |
| `training_history.csv` | Perdida y metricas por epoca. |
| `valid_confusion_matrix.csv` | Matriz de confusion de validacion. |
| `test_confusion_matrix.csv` | Matriz de confusion de prueba. |
| `class_names.json` | Lista de clases del modelo. |
| `README.md` | Resumen automatico de resultados. |

## Metricas usadas

Como el modelo es de clasificacion, las metricas principales son:

| Metrica | Que mide |
|---|---|
| Accuracy | Porcentaje total de recortes clasificados correctamente. |
| Macro precision | Promedio de precision por clase. Penaliza falsos positivos. |
| Macro recall | Promedio de recall por clase. Penaliza falsos negativos. |
| Macro F1-score | Balance entre precision y recall por clase. Es la metrica principal recomendada. |
| Matriz de confusion | Permite ver que piezas se confunden entre si. |

DSC/Dice no es la metrica principal de este modelo, porque este flujo clasifica recortes de dientes. Dice seria mas adecuado si el resultado final fuera una mascara de segmentacion.

## Resultado tecnico ya comprobado

Se ejecuto una prueba rapida con 12 radiografias y 1 epoca para comprobar que el pipeline funciona de punta a punta. Esta prueba no representa el rendimiento final del modelo; solo valida que el codigo puede:

- preparar recortes desde poligonos;
- construir el dataset;
- entrenar la CNN;
- guardar checkpoints;
- generar metricas;
- escribir matrices de confusion.

Resultado de esa prueba tecnica:

| Elemento | Valor |
|---|---:|
| Radiografias usadas | 12 |
| Recortes train | 194 |
| Recortes valid | 21 |
| Recortes test | 48 |
| Epocas | 1 |
| Test accuracy | 0.0208 |
| Test macro F1 | 0.0013 |

Estos valores son bajos porque fue una prueba minima de ejecucion, no un entrenamiento real. No deben presentarse como resultado final del proyecto.

## Resultado que falta obtener

Falta ejecutar el entrenamiento completo con todas las muestras preparadas. Ese entrenamiento deberia reportarse como resultado final del proyecto.

Comando recomendado:

```powershell
python src/train_tooth_piece_cnn.py `
  --skip-prepare `
  --epochs 20 `
  --batch-size 64 `
  --image-size 128 `
  --device cpu
```

Despues de ejecutarlo, los resultados que se deben revisar son:

```text
models/tooth_piece_cnn_classifier/metrics.json
models/tooth_piece_cnn_classifier/training_history.csv
models/tooth_piece_cnn_classifier/test_confusion_matrix.csv
```

## Lectura para presentacion

El flujo del proyecto parte desde un dataset de radiografias dentales con anotaciones por poligono. Cada poligono corresponde a una pieza dental, por lo que se utiliza para aislar el diente y construir un recorte individual con una etiqueta clara. A partir de esos recortes se entrena una CNN supervisada que aprende patrones visuales de cada pieza y predice una de 32 clases.

La ventaja de este flujo es que usa directamente las anotaciones disponibles, evita clasificar radiografias completas con multiples dientes y genera metricas de clasificacion interpretables. El resultado final esperado es un modelo local `best.pt`, acompanado por accuracy, macro precision, macro recall, macro F1-score y matriz de confusion.
