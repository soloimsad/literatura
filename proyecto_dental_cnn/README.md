# Proyecto CNN para segmentacion y rotulado dental

Este proyecto implementa un pipeline en Python para recibir una radiografia
panoramica dental y generar una imagen de salida con regiones segmentadas y
rotulos. La idea sigue la metodologia de los papers revisados: una CNN hace la
segmentacion semantica inicial y despues se aplican operaciones de imagen para
separar instancias, limpiar ruido y dibujar la salida interpretable.

## Relacion con la literatura

- Papers 2 y 3: motivan la segmentacion automatica de dientes y el problema de
  separar piezas individuales cuando hay dientes faltantes, apiñamiento o
  protesis.
- Paper 4: usa CNN para segmentacion semantica y luego procesamiento de imagen
  para obtener regiones, cajas y etiquetas.
- Paper 5: respalda el flujo practico con Python/CNN para extraer informacion
  visual desde imagenes.
- Paper 7: justifica el uso de GPU/CUDA cuando este disponible, aunque el codigo
  tambien corre en CPU para pruebas chicas.

Los datasets de Kaggle entregan principalmente cajas de anotacion, no mascaras
finas. Por eso el entrenamiento crea mascaras debiles desde las cajas y la etapa
de inferencia refina el resultado con morfologia y componentes conectados.

## Instalacion

Desde esta carpeta:

```powershell
python -m pip install -r requirements.txt
```

Si quieres instalar PyTorch CPU explicitamente:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Para CUDA/NVIDIA conviene usar el comando recomendado por PyTorch para tu version
de CUDA y despues correr el entrenamiento. El script selecciona `cuda` si
`torch.cuda.is_available()` devuelve verdadero.

## Descargar datasets

```powershell
python -m dental_cnn.download_datasets
```

Tambien puedes pasar las rutas descargadas manualmente con `--data-root`.

## Entrenar

Entrenamiento rapido de prueba:

```powershell
python -m dental_cnn.train --mask-mode binary --epochs 1 --batch-size 2 --max-train-samples 64 --max-valid-samples 16 --output-dir runs/smoke
```

Entrenamiento mas defendible:

```powershell
python -m dental_cnn.train --epochs 15 --batch-size 4 --image-width 512 --image-height 256 --output-dir runs/dental_unet
```

Entrenamiento enfocado en algunas clases:

```powershell
python -m dental_cnn.train --mask-mode multiclass --include-class Caries --include-class Filling --include-class Implant --epochs 10 --output-dir runs/dental_focused
```

El checkpoint queda en:

```text
runs/dental_unet/best_model.pt
```

## Predecir una radiografia

```powershell
python -m dental_cnn.predict ruta\a\radiografia.jpg --checkpoint runs/dental_unet/best_model.pt --output outputs\rotulada.png
```

La salida incluye:

- `outputs/rotulada.png`: imagen con mascara, contornos, cajas y rotulos.
- `outputs/rotulada.json`: lista de instancias con clase, caja, area y score.

Puedes usar rotulado secuencial (`T01`, `T02`, ...) o aproximacion FDI:

```powershell
python -m dental_cnn.predict ruta\a\radiografia.jpg --checkpoint runs/dental_unet/best_model.pt --label-style fdi
```

## Notas para presentar

El proyecto no afirma diagnostico clinico. Es un prototipo academico de
segmentacion/rotulado basado en CNN, entrenado con anotaciones publicas. La
principal limitacion es que algunas mascaras son aproximadas porque los datasets
proveen cajas, no contornos manuales de cada diente.
