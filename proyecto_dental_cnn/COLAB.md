# Guia rapida para Google Colab

Esta es la ruta recomendada para trabajar el proyecto, porque el entrenamiento
con imagenes medicas se beneficia harto de GPU.

## 1. Subir el proyecto a Drive

Sube la carpeta completa:

```text
proyecto_dental_cnn
```

a:

```text
Mi unidad/proyecto_dental_cnn
```

La ruta esperada por el notebook es:

```text
/content/drive/MyDrive/proyecto_dental_cnn
```

## 2. Abrir notebook

Abre en Colab:

```text
notebooks/proyecto_dental_cnn_colab.ipynb
```

Luego cambia el entorno:

```text
Entorno de ejecucion > Cambiar tipo de entorno de ejecucion > GPU
```

## 3. Entrenamiento recomendado

El notebook ya trae este comando:

```bash
python -m dental_cnn.train \
  --mask-mode binary \
  --epochs 20 \
  --batch-size 16 \
  --image-width 512 \
  --image-height 256 \
  --base-channels 16 \
  --num-workers 2 \
  --amp \
  --output-dir /content/drive/MyDrive/dental_cnn_runs/binary_unet
```

Esto guarda el mejor modelo en Drive:

```text
/content/drive/MyDrive/dental_cnn_runs/binary_unet/best_model.pt
```

## 4. Por que binario primero

Los datasets descargados desde Kaggle tienen principalmente cajas de anotacion,
no mascaras finas dibujadas a mano. Para una entrega estable conviene entrenar
primero una U-Net binaria: region dental/anotada versus fondo. Despues el
programa separa regiones con post-procesamiento y dibuja los rotulos.

Si quieres mostrar clasificacion de problemas dentales, el notebook tambien
incluye una celda multiclasica opcional para clases como `Caries`, `Filling`,
`Implant`, `Crown`, `Root Canal Treatment`, `impacted tooth` y `Missing teeth`.

## 5. Prediccion

Con un checkpoint ya entrenado:

```bash
python -m dental_cnn.predict radiografia.jpg \
  --checkpoint /content/drive/MyDrive/dental_cnn_runs/binary_unet/best_model.pt \
  --output /content/drive/MyDrive/dental_cnn_runs/predictions/rotulada.png \
  --label-style fdi
```

La salida incluye:

- Imagen `.png` con mascaras, contornos, cajas y rotulos.
- Archivo `.json` con clase, caja, area y score por instancia.

## 6. Frase honesta para la presentacion

El prototipo no entrega diagnostico clinico. Es una demostracion academica de
segmentacion y rotulado con CNN, entrenada con anotaciones publicas. Como las
mascaras provienen de cajas, el resultado debe presentarse como segmentacion
debil/refinada por procesamiento de imagen.
