# Metricas del modelo de clasificacion de piezas dentales

Estas metricas corresponden al modelo local:

```text
models/tooth_piece_classifier/weights/best.pt
```

El modelo es un **YOLO-seg** para deteccion, segmentacion y clasificacion de piezas dentales. Las metricas fueron extraidas desde el checkpoint local del modelo `tooth`.

## Datos del checkpoint

| Item | Valor |
|---|---|
| Tarea | Segmentacion |
| Modelo base | `yolo11x-seg.pt` |
| Clases | 54 |
| Imagen de entrenamiento | 1280 px |
| Batch | 4 |
| Epocas configuradas | 100 |
| Epocas registradas | 56 |
| Fecha del checkpoint | 2025-05-08 |

## Metricas principales

| Tipo | Precision | Recall | F1-score | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Bounding boxes | 0.9258 | 0.8339 | 0.8775 | 0.9004 | 0.7670 |
| Mascaras | 0.9332 | 0.8412 | 0.8848 | 0.9122 | 0.7344 |

## Metricas en porcentaje

| Tipo | Precision | Recall | F1-score | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Bounding boxes | 92.58% | 83.39% | 87.75% | 90.04% | 76.70% |
| Mascaras | 93.32% | 84.12% | 88.48% | 91.22% | 73.44% |

## Slide sugerida

**Resultados del modelo de piezas dentales**

- Modelo YOLO-seg entrenado para clasificar y segmentar piezas dentales.
- 54 clases dentales.
- Precision en mascaras: **93.32%**.
- Recall en mascaras: **84.12%**.
- F1-score en mascaras: **88.48%**.
- mAP50 en mascaras: **91.22%**.

## Como explicarlo oralmente

> El modelo de piezas dentales fue evaluado como un modelo YOLO-seg, por lo que no solo se mide si clasifica correctamente la pieza, sino tambien si la localiza y segmenta adecuadamente. Para las mascaras, el modelo alcanza una precision de 93.32%, un recall de 84.12% y un F1-score de 88.48%. Ademas, obtiene un mAP50 de 91.22%, lo que indica un buen desempeno en la deteccion/segmentacion de piezas dentales.

## Nota tecnica

El F1-score fue calculado a partir de precision y recall guardados en el checkpoint. DSC/Dice no viene almacenado directamente en el modelo; para calcularlo seria necesario volver a validar contra mascaras ground truth compatibles.
