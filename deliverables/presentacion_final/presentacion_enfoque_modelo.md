# Presentacion: Justificacion del enfoque de modelado

## Diapositiva 1 - Titulo

**Clasificacion de piezas dentales usando deep learning**

Proyecto final INF-402  
Procesamiento y Reconstruccion de Imagenes Medicas

**Idea clave:**  
Entrenar un modelo capaz de clasificar piezas dentales individuales a partir de radiografias anotadas.

**Notas para exponer:**  
En esta presentacion voy a explicar por que elegimos este enfoque de modelado. La idea principal no es clasificar una radiografia completa, sino reconocer piezas dentales individuales. Para eso aprovechamos que el dataset tiene anotaciones por poligono, lo que permite aislar cada pieza y entrenar un modelo supervisado.

---

## Diapositiva 2 - Problema

**Una radiografia panoramica no contiene una sola clase**

- Una imagen puede contener muchas piezas dentales.
- Cada pieza tiene una identidad distinta.
- Una etiqueta global para toda la imagen no sirve para este objetivo.
- El modelo debe aprender pieza por pieza.

**Notas para exponer:**  
El problema es que una radiografia panoramica tiene multiples dientes visibles. Si entrenaramos una CNN con la imagen completa y una sola etiqueta, el modelo no sabria que pieza especifica debe clasificar. Por eso necesitamos transformar el problema: en vez de clasificar la radiografia completa, clasificamos cada pieza dental individualmente.

---

## Diapositiva 3 - Objetivo del modelo

**Objetivo**

Clasificar recortes individuales de piezas dentales en una de las clases del dataset.

**Entrada del modelo:**  
Recorte de una pieza dental.

**Salida del modelo:**  
Clase de la pieza dental.

**Metrica principal:**  
Macro F1-score.

**Notas para exponer:**  
El modelo recibe como entrada una imagen pequena que contiene una pieza dental aislada. A partir de ese recorte, la CNN predice la clase correspondiente. Usamos macro F1-score porque hay muchas clases y no basta con mirar solo accuracy; necesitamos evaluar el rendimiento de forma balanceada entre clases.

---

## Diapositiva 4 - Por que no usar la radiografia completa

**Clasificar la imagen completa seria ambiguo**

- Una radiografia contiene multiples dientes.
- Las piezas estan muy cerca entre si.
- Algunas piezas comparten formas similares.
- La posicion y el contexto influyen.
- Una sola etiqueta perderia informacion anatomica.

**Notas para exponer:**  
La radiografia completa tiene demasiada informacion mezclada. El modelo podria aprender patrones generales de la imagen, pero no necesariamente aprenderia a distinguir cada pieza. Como el objetivo es clasificar piezas dentales, necesitamos que cada muestra del dataset represente una pieza clara y no una escena completa con muchos objetos.

---

## Diapositiva 5 - Por que usar poligonos

**Los poligonos son la supervision mas valiosa del dataset**

- Marcan la region exacta de cada diente.
- Permiten separar pieza y fondo.
- Generan una muestra individual por pieza.
- Reducen ruido visual.
- Aprovechan mejor las anotaciones disponibles.

**Notas para exponer:**  
Los poligonos son importantes porque indican donde esta cada pieza dental. A diferencia de una caja rectangular, el poligono sigue mejor el contorno del diente. Esto permite crear una mascara, aislar la pieza y generar un recorte limpio para entrenar. Asi usamos la informacion mas precisa que trae el dataset.

---

## Diapositiva 6 - Transformacion del dataset

**De radiografias completas a recortes dentales**

1. Cargar radiografia.
2. Leer anotacion JSON.
3. Extraer poligono de cada pieza.
4. Crear mascara.
5. Recortar el diente.
6. Asignar etiqueta.
7. Guardar recorte por clase.

**Notas para exponer:**  
El flujo de preparacion convierte cada radiografia en muchas muestras de entrenamiento. Cada poligono se transforma en un recorte individual y hereda la clase de la anotacion original. De esta forma pasamos de un dataset de radiografias completas a un dataset de clasificacion supervisada por pieza dental.

---

## Diapositiva 7 - Datos preparados

**Resultado de preparacion**

| Split | Radiografias | Recortes dentales |
|---|---:|---:|
| Train | 478 | 12198 |
| Valid | 59 | 1518 |
| Test | 61 | 1602 |
| Total | 598 | 15318 |

**Clases:** 32 piezas dentales.

**Notas para exponer:**  
Despues de procesar los poligonos, se obtuvieron 15.318 recortes dentales. El split se hizo por radiografia, no por recorte. Esto es importante porque evita que dientes de una misma imagen aparezcan en entrenamiento y validacion al mismo tiempo, lo que podria inflar artificialmente los resultados.

---

## Diapositiva 8 - Por que una CNN

**Las CNN aprenden patrones visuales locales**

- Bordes.
- Curvaturas.
- Texturas.
- Cambios de intensidad.
- Forma de corona y raiz.
- Diferencias morfologicas entre piezas.

**Notas para exponer:**  
Elegimos una CNN porque las piezas dentales se diferencian por su forma y estructura visual. Las convoluciones son adecuadas para aprender patrones espaciales, como bordes, curvas y texturas. En capas mas profundas, la red combina esos patrones para construir una representacion mas compleja de cada pieza.

---

## Diapositiva 9 - Arquitectura implementada

**Modelo: ToothPieceCNN**

- Entrada en escala de grises.
- Bloques convolucionales 2D.
- Batch Normalization.
- ReLU.
- MaxPooling.
- Adaptive Average Pooling.
- Capas fully connected.
- Salida multiclase.

**Notas para exponer:**  
El modelo implementado se llama ToothPieceCNN. Es una CNN propia hecha en PyTorch. No es un modelo externo ni un servicio remoto. La red recibe recortes en escala de grises, aprende caracteristicas con capas convolucionales y finalmente clasifica cada pieza dental mediante una salida multiclase entrenada con CrossEntropyLoss.

---

## Diapositiva 10 - Relacion con la literatura

**La decision se apoya en tres ideas de los papers**

| Paper | Aporte al proyecto |
|---|---|
| Tooth Segmentation in 3D CBCT Using CNN | Las CNN aprenden estructuras dentales. |
| Fully Automated 3D Individual Tooth Identification and Segmentation | La identificacion individual de piezas es una tarea relevante. |
| Teeth Detection and Dental Problem Classification in Panoramic X-Ray | Deep learning puede aplicarse a radiografias panoramicas dentales. |

**Notas para exponer:**  
La literatura revisada respalda la decision de trabajar con estructuras dentales individuales. El paper de segmentacion en CBCT muestra que las CNN son utiles para aprender regiones dentales. El paper de identificacion individual refuerza que el objetivo no es solo detectar dientes, sino distinguir piezas. Y el paper sobre panoramicas conecta este tipo de enfoque con radiografias 2D.

---

## Diapositiva 11 - Por que esta decision es coherente

**Coherencia entre problema, datos y modelo**

- Problema: clasificar piezas individuales.
- Dataset: poligonos por pieza dental.
- Modelo: CNN para clasificacion visual.
- Evaluacion: metricas de clasificacion multiclase.

**Notas para exponer:**  
La decision metodologica es coherente porque el modelo se adapta al dataset y al objetivo. Si tenemos poligonos por pieza, lo natural es usarlos para construir muestras individuales. Si el objetivo es clasificar piezas, lo natural es entrenar un clasificador multiclase. Y si la entrada son imagenes, una CNN es una opcion adecuada.

---

## Diapositiva 12 - Metricas de evaluacion

**Metricas usadas**

- Accuracy.
- Macro precision.
- Macro recall.
- Macro F1-score.
- Matriz de confusion.

**Por que macro F1:**  
Evalua el rendimiento promedio por clase y es mas informativa cuando hay multiples clases.

**Notas para exponer:**  
Usamos metricas de clasificacion porque el modelo predice clases de piezas dentales. Macro F1 es especialmente importante porque promedia el rendimiento entre clases. Eso evita que una clase con muchas muestras domine la evaluacion. La matriz de confusion tambien es util porque permite ver que piezas se confunden entre si.

---

## Diapositiva 13 - Resultado tecnico actual

**Lo que ya esta implementado**

- Preparacion completa del dataset.
- Generacion de recortes desde poligonos.
- Split train/valid/test.
- CNN en PyTorch.
- Entrenamiento reproducible.
- Guardado de modelo y metricas.

**Prueba tecnica:**  
Se ejecuto una prueba pequena de 1 epoca para validar que el pipeline funciona.

**Notas para exponer:**  
Actualmente el flujo ya esta implementado y probado tecnicamente. La prueba pequena no se presenta como resultado final, porque solo valida que el codigo corre de punta a punta. Lo importante es que el pipeline ya prepara datos, entrena la red, guarda checkpoints y genera metricas.

---

## Diapositiva 14 - Resultado esperado

**Al ejecutar el entrenamiento completo se obtiene**

- `best.pt`: mejor modelo.
- `metrics.json`: metricas finales.
- `training_history.csv`: evolucion del entrenamiento.
- `test_confusion_matrix.csv`: errores por clase.
- `class_names.json`: clases del modelo.

**Notas para exponer:**  
El resultado final del entrenamiento completo sera un modelo local y reproducible. Ademas del peso entrenado, se guardan metricas y matrices de confusion para analizar el rendimiento. Esto permite presentar resultados cuantitativos y tambien discutir limitaciones, como clases que el modelo confunde.

---

## Diapositiva 15 - Mensaje final

**Decision metodologica principal**

Usamos los poligonos para transformar radiografias completas en recortes individuales de dientes, y entrenamos una CNN para clasificar cada pieza.

**Por que esta bien elegido**

- Aprovecha la anotacion mas precisa del dataset.
- Evita una clasificacion global ambigua.
- Se alinea con el objetivo del proyecto.
- Es coherente con la literatura revisada.
- Produce metricas claras de clasificacion.

**Notas para exponer:**  
La decision central fue adaptar el modelo a la forma real del problema. Como queremos clasificar piezas, no radiografias completas, usamos los poligonos para generar recortes individuales. Esto hace que el entrenamiento sea mas claro, mas controlado y mas facil de evaluar. Por eso esta estrategia es defendible tanto desde el punto de vista tecnico como desde la literatura.

