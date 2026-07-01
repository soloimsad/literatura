# Posible tratamiento o hallazgo

Modelo local esperado para detectar hallazgos dentales asociados a posibles
tratamientos, por ejemplo caries, coronas, restauraciones, implantes o lesiones.

Ruta esperada:

```text
models/treatment_detector/weights/best.pt
```

El archivo `best.pt` existe localmente en esta carpeta y es usado por los
notebooks y `src/predict_dental_xray.py` mediante la registry del proyecto.
