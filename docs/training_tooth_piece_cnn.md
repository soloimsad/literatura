# Entrenamiento deep learning para clasificacion de piezas dentales

Este documento describe el nuevo flujo de entrenamiento del modelo deep learning para el proyecto final, enfocado exclusivamente en **clasificacion de piezas dentales**.

## Idea principal

El dataset disponible contiene radiografias dentales con anotaciones por poligono. Cada poligono marca una pieza dental y trae una clase asociada. En vez de usar directamente un modelo externo ya entrenado, el nuevo flujo entrena una CNN propia a partir de esos datos.

El procedimiento es:

1. Leer las radiografias y los JSON de anotaciones en formato Supervisely.
2. Extraer cada poligono dental.
3. Convertir cada poligono en un recorte individual de diente.
4. Usar la clase del poligono como etiqueta del recorte.
5. Dividir las radiografias en train, valid y test.
6. Entrenar una CNN en PyTorch para clasificar cada recorte dental.
7. Guardar metricas, matriz de confusion y el mejor modelo.

## Por que usar poligonos

Los poligonos son utiles porque entregan una supervision mas precisa que una etiqueta global. En este proyecto no queremos decir solamente "esta radiografia tiene dientes", sino aprender a reconocer piezas individuales. El poligono permite aislar cada pieza dental, generar un recorte limpio y entrenar el modelo con una muestra por diente.

Esto tambien evita entrenar con la radiografia completa, que contiene muchas piezas al mismo tiempo. Para una CNN clasificadora, cada muestra debe tener una etiqueta clara. Por eso se transforma cada diente anotado en una imagen individual.

## Codigo principal

El script nuevo esta en:

```text
src/train_tooth_piece_cnn.py
```

Este script prepara los datos y entrena el modelo. No usa el modelo de tratamientos y no depende de Google Drive.

## Dataset generado

La preparacion completa ya fue ejecutada y genero:

```text
data/_prepared_tooth_piece_classifier/
```

Resumen del dataset preparado:

| Split | Radiografias | Recortes dentales |
|---|---:|---:|
| Train | 478 | 12198 |
| Valid | 59 | 1518 |
| Test | 61 | 1602 |

El split se realiza por radiografia, no por recorte. Esto es importante porque evita que dientes de una misma radiografia queden mezclados entre entrenamiento y validacion.

## Arquitectura

El modelo implementado es una CNN propia llamada `ToothPieceCNN`. Usa:

- entrada en escala de grises;
- bloques convolucionales 2D;
- BatchNorm;
- ReLU;
- MaxPooling;
- Adaptive Average Pooling;
- capas fully connected;
- salida softmax implicita mediante CrossEntropyLoss.

No es una CNN preentrenada ni una llamada a un servicio externo. Es un modelo deep learning entrenable localmente con PyTorch.

## Comandos

Preparar datos:

```powershell
python src/train_tooth_piece_cnn.py --prepare-only --force-prepare
```

Entrenar modelo completo:

```powershell
python src/train_tooth_piece_cnn.py `
  --skip-prepare `
  --epochs 20 `
  --batch-size 64 `
  --image-size 128 `
  --device cpu
```

Si hay GPU disponible, se puede usar:

```powershell
python src/train_tooth_piece_cnn.py `
  --skip-prepare `
  --epochs 20 `
  --batch-size 64 `
  --image-size 128 `
  --device cuda
```

Prueba rapida con pocas radiografias:

```powershell
python src/train_tooth_piece_cnn.py `
  --limit-images 12 `
  --prepared-dir results/smoke_tooth_classifier_data `
  --output-dir results/smoke_tooth_classifier_model `
  --force-prepare `
  --image-size 96 `
  --epochs 1 `
  --batch-size 16 `
  --device cpu
```

## Salidas del entrenamiento

Por defecto el entrenamiento completo guarda resultados en:

```text
models/tooth_piece_cnn_classifier/
```

Archivos principales:

- `best.pt`: mejor modelo segun macro F1 de validacion.
- `last.pt`: ultimo modelo entrenado.
- `metrics.json`: metricas finales en test.
- `training_history.csv`: perdida y metricas por epoca.
- `valid_confusion_matrix.csv`: matriz de confusion de validacion.
- `test_confusion_matrix.csv`: matriz de confusion de prueba.
- `class_names.json`: clases aprendidas por el modelo.

## Metricas

El script calcula:

- accuracy;
- macro precision;
- macro recall;
- macro F1-score;
- matriz de confusion.

Estas metricas son coherentes con la pauta para una tarea de clasificacion. DSC/Dice no corresponde como metrica principal de este modelo, porque el modelo final clasifica recortes dentales. Dice seria mas adecuado para evaluar segmentacion de mascaras.

## Explicacion breve para presentacion

El modelo se entrena usando las anotaciones por poligono del dataset dental. Cada poligono representa una pieza dental individual, por lo que se usa para recortar la pieza desde la radiografia y generar una muestra de entrenamiento con una etiqueta clara. Luego, esos recortes se dividen en entrenamiento, validacion y prueba, cuidando que la division sea por radiografia para evitar fuga de informacion entre splits.

Sobre esos recortes se entrena una CNN en PyTorch. La red aprende patrones visuales de cada pieza dental mediante capas convolucionales, y al final entrega una clase para cada recorte. Este enfoque es deep learning supervisado y se relaciona con los papers de segmentacion e identificacion dental porque usa las regiones dentales anotadas para aprender representaciones visuales de piezas individuales.
