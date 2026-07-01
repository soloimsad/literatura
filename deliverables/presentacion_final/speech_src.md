# Speech para explicar la carpeta `src/`

Este archivo esta separado por pestañas de explicacion. Cada pestaña puede usarse como bloque de discurso para una diapositiva o como guia para responder preguntas tecnicas.

## Pestaña 1 - Vision general de `src/`

En la carpeta `src/` se encuentra el nucleo tecnico del proyecto. Esta carpeta concentra el codigo que permite preparar los datos, definir los modelos disponibles y ejecutar inferencia sobre radiografias dentales.

La organizacion esta pensada para separar responsabilidades. No todo el codigo hace lo mismo: hay un archivo dedicado a registrar los modelos locales, otro dedicado a preparar datos y eventualmente entrenar, y otro dedicado a ejecutar la demostracion final sobre imagenes.

En terminos generales, el flujo es el siguiente:

1. Se preparan y auditan los datasets.
2. Se definen los modelos locales disponibles.
3. Se ejecuta inferencia sobre una radiografia.
4. Se generan resultados visuales y reportes CSV.

Esta separacion ayuda a que el proyecto sea mas reproducible y mas facil de explicar, porque cada archivo tiene un rol claro dentro del pipeline.

## Pestaña 2 - `model_registry.py`

El archivo `model_registry.py` funciona como una registry o registro central de modelos. Su objetivo es declarar que modelos existen en el proyecto, donde estan ubicados sus pesos entrenados y que rol cumple cada uno.

En este proyecto se definen dos modelos principales:

- `tooth`: modelo de clasificacion y segmentacion de piezas dentales.
- `treatment`: modelo de deteccion de posibles tratamientos o hallazgos dentales.

El modelo `tooth` apunta a:

```text
models/tooth_piece_classifier/weights/best.pt
```

El modelo `treatment` apunta a:

```text
models/treatment_detector/weights/best.pt
```

La importancia de este archivo es que evita tener rutas escritas manualmente en distintas partes del codigo. En vez de que cada script tenga su propia ruta, todos consultan la misma fuente de verdad.

Ademas, esta registry ayuda a garantizar que el proyecto use modelos locales dentro de la carpeta `models/`. Esto es importante para la reproducibilidad, porque el sistema no depende de descargar modelos externos ni de rutas personales del equipo.

Una forma simple de explicarlo seria:

> `model_registry.py` es el archivo que le dice al proyecto que modelos existen, donde estan y para que se usan.

## Pestaña 3 - `dental_xray_pipeline.py`

El archivo `dental_xray_pipeline.py` corresponde al pipeline de preparacion de datos y entrenamiento. Su funcion principal es tomar los datasets disponibles, revisar su estructura, auditar sus anotaciones y preparar los archivos necesarios para trabajar con YOLO.

Este archivo permite trabajar con dos tipos de datos:

- Dataset de piezas dentales.
- Dataset de hallazgos o posibles tratamientos.

Para el dataset de piezas dentales, el pipeline toma anotaciones poligonales y las convierte a un formato compatible con YOLO-seg. Esto permite que el modelo no solo aprenda a clasificar piezas dentales, sino tambien a segmentarlas dentro de la radiografia.

Para el dataset de tratamientos o hallazgos, el pipeline revisa la estructura YOLO del dataset, identifica las clases disponibles y genera archivos de configuracion como `data.yaml`.

El pipeline tambien genera archivos de auditoria, por ejemplo:

- cantidad de imagenes;
- cantidad de anotaciones;
- clases disponibles;
- distribucion de objetos;
- resumen de preparacion.

Esto es util para la parte de materiales y metodos, porque permite demostrar que los datos fueron revisados y preparados de manera reproducible.

Una explicacion natural seria:

> `dental_xray_pipeline.py` es el archivo que prepara el terreno. Antes de mostrar resultados o entrenar modelos, este script revisa los datasets, los transforma al formato necesario y deja evidencia de esa preparacion.

## Pestaña 4 - `predict_dental_xray.py`

El archivo `predict_dental_xray.py` es el script principal para la demostracion del sistema. Este archivo toma una radiografia como entrada, carga uno o ambos modelos locales y genera resultados visuales.

Puede ejecutarse con tres modos:

```text
--model-role tooth
--model-role treatment
--model-role both
```

Con `tooth`, el sistema detecta y clasifica piezas dentales. El resultado visual muestra las piezas rotuladas con etiquetas tipo FDI.

Con `treatment`, el sistema detecta hallazgos dentales asociados a posibles tratamientos, como caries, coronas, restauraciones, implantes u otras condiciones.

Con `both`, se ejecutan ambos modelos sobre la misma radiografia.

Este script genera dos tipos de salida:

- una imagen rotulada;
- un archivo CSV con las predicciones.

La imagen sirve para la demostracion visual durante la presentacion. El CSV sirve como respaldo tecnico, porque contiene informacion como clase predicha, confianza y coordenadas.

Una forma clara de decirlo seria:

> `predict_dental_xray.py` es el archivo que usamos para la demo. Recibe una radiografia, aplica el modelo correspondiente y entrega una imagen marcada junto con un reporte de predicciones.

## Pestaña 5 - Que tipo de red se usa

Los modelos utilizados son modelos YOLO de segmentacion, especificamente modelos tipo YOLO-seg.

YOLO significa "You Only Look Once". Es una familia de arquitecturas de vision por computador orientada a deteccion de objetos. A diferencia de un clasificador tradicional, que recibe una imagen completa y entrega una sola etiqueta, YOLO puede detectar multiples objetos dentro de una misma imagen.

En este proyecto eso es importante porque una radiografia panoramica contiene muchas estructuras al mismo tiempo. No basta con decir que la imagen tiene dientes o que puede existir un hallazgo. Lo importante es localizar donde esta cada pieza dental o cada posible condicion.

YOLO entrega informacion como:

- clase predicha;
- confianza;
- caja delimitadora;
- mascara de segmentacion, cuando aplica.

Al usar YOLO-seg, el modelo puede aproximar el contorno del objeto detectado, no solo encerrarlo en una caja.

## Pestaña 6 - Es una CNN?

Si, la base de este tipo de modelo esta relacionada con redes neuronales convolucionales, o CNN.

Una CNN aprende filtros visuales que permiten detectar patrones en imagenes. En una radiografia dental, esos patrones pueden ser bordes, formas, texturas, contraste entre estructuras, limites de piezas dentales, restauraciones o zonas asociadas a posibles lesiones.

Sin embargo, es importante explicarlo con precision: no estamos usando una CNN clasica de clasificacion simple. Una CNN clasica podria tomar una imagen completa y decir solamente "clase A" o "clase B".

En este proyecto se usa una arquitectura moderna de deteccion y segmentacion basada en CNN. Es decir, las capas convolucionales ayudan a extraer caracteristicas visuales, y luego el modelo usa esas caracteristicas para predecir objetos, clases, ubicaciones y mascaras.

Una frase tecnica correcta seria:

> El proyecto usa modelos YOLO-seg, que son arquitecturas modernas de deteccion y segmentacion basadas en redes convolucionales. Por lo tanto, no es una CNN clasica de clasificacion global, sino un modelo de vision por computador para deteccion y segmentacion de multiples objetos.

## Pestaña 7 - Modelo de pieza dental

El modelo de pieza dental corresponde al rol `tooth`.

Su objetivo es identificar piezas dentales individuales dentro de una radiografia panoramica. El modelo entrega una clase para cada pieza detectada, usando etiquetas asociadas a numeracion dental.

Este modelo permite mostrar visualmente donde esta cada pieza y que numero se le asigna. En una presentacion, esto es util porque transforma una radiografia dificil de interpretar en una imagen anotada, donde se pueden ver las predicciones del sistema.

La salida principal es una imagen como:

```text
sample-19_tooth_piece_classifier_labeled.jpg
```

Y un reporte como:

```text
sample-19_tooth_piece_classifier_report.csv
```

El CSV contiene las detecciones, la confianza del modelo y las coordenadas. Esto permite complementar la visualizacion con datos mas estructurados.

## Pestaña 8 - Modelo de tratamiento o hallazgos

El modelo de tratamiento corresponde al rol `treatment`.

Su objetivo es detectar hallazgos dentales que pueden estar asociados a tratamientos o condiciones clinicas. Algunas clases son, por ejemplo:

- Caries;
- Crown;
- Filling;
- Implant;
- Missing teeth;
- Periapical lesion;
- Root Canal Treatment;
- Bone Loss.

Este modelo no busca numerar piezas dentales, sino detectar elementos clinicos o restaurativos visibles en la radiografia.

En la presentacion, este modelo se puede explicar como una segunda etapa del sistema: primero se puede localizar y clasificar piezas dentales, y luego se pueden detectar hallazgos relevantes para apoyar el analisis odontologico.

La salida principal es una imagen como:

```text
sample-19_treatment_detector_labeled.jpg
```

Y un reporte como:

```text
sample-19_treatment_detector_report.csv
```

## Pestaña 9 - Flujo completo del proyecto

El flujo completo puede explicarse en cuatro etapas.

Primero, los datos se preparan con `dental_xray_pipeline.py`. En esta etapa se revisan las imagenes, las anotaciones y las clases del dataset. Tambien se generan archivos compatibles con YOLO.

Segundo, los modelos se declaran en `model_registry.py`. Esta etapa centraliza las rutas y evita depender de modelos externos.

Tercero, la inferencia se ejecuta con `predict_dental_xray.py`. Ahi se carga la radiografia de entrada y se aplica el modelo elegido.

Cuarto, se generan los resultados. Estos resultados son principalmente imagenes rotuladas y reportes CSV.

Una forma simple de presentarlo seria:

> El proyecto sigue un flujo reproducible: preparo datos, uso modelos locales, ejecuto inferencia y genero resultados visuales y tabulares.

## Pestaña 10 - Relacion con la evaluacion

Segun los lineamientos del curso, el proyecto debe ser reproducible, debe incluir codigo claro, resultados visuales, metricas o reportes, y una demostracion practica.

La carpeta `src/` apoya directamente esos criterios.

`dental_xray_pipeline.py` aporta a la reproducibilidad y a la explicacion de materiales y metodos.

`model_registry.py` aporta al orden del proyecto, porque centraliza los modelos locales.

`predict_dental_xray.py` aporta a la demostracion practica, porque permite generar imagenes y reportes desde consola.

Los resultados generados se pueden usar en la presentacion final como evidencia visual y como respaldo tecnico.

Una frase de cierre seria:

> La organizacion de `src/` permite conectar la parte metodologica con la demostracion. No solo tenemos modelos entrenados, sino tambien un flujo reproducible para preparar datos, cargar modelos locales y generar resultados interpretables.

## Pestaña 11 - Version fluida para decir en voz alta

En la carpeta `src/` esta el nucleo tecnico del proyecto. Esta carpeta esta dividida en tres partes principales: una registry de modelos, un pipeline de preparacion de datos y un script de inferencia.

La registry, que esta en `model_registry.py`, define que modelos existen y donde estan sus pesos locales. Esto es importante porque permite que el proyecto sea reproducible y no dependa de modelos descargados desde internet o rutas externas.

El pipeline, que esta en `dental_xray_pipeline.py`, se encarga de preparar los datasets. Revisa las imagenes, las anotaciones, las clases y genera archivos compatibles con YOLO. Tambien produce reportes de auditoria que sirven para explicar la metodologia.

Finalmente, `predict_dental_xray.py` es el archivo que usamos para la demostracion. Este script recibe una radiografia, ejecuta el modelo de pieza dental, el modelo de tratamiento o ambos, y genera una imagen rotulada junto con un archivo CSV.

Los modelos usados son YOLO de segmentacion. Esto significa que no son clasificadores simples, sino modelos que detectan multiples objetos dentro de una imagen. Ademas, al ser YOLO-seg, pueden entregar mascaras o contornos aproximados, no solamente cajas.

Desde el punto de vista de arquitectura, YOLO esta basado en redes convolucionales. Entonces si, utiliza CNN como base para extraer caracteristicas visuales, pero no es una CNN clasica de clasificacion global. Es una arquitectura moderna de deteccion y segmentacion basada en CNN.

En resumen, el proyecto usa modelos locales YOLO-seg para analizar radiografias dentales. Un modelo identifica piezas dentales y otro detecta posibles tratamientos o hallazgos. El resultado final son imagenes interpretables y reportes estructurados que pueden usarse en la presentacion y en el analisis del proyecto.

## Pestaña 12 - Respuesta corta para preguntas

Si preguntan que red se uso:

> Se uso YOLO-seg, una arquitectura de deteccion y segmentacion basada en redes convolucionales.

Si preguntan si es CNN:

> Si, la base de YOLO usa convoluciones para extraer caracteristicas visuales, pero no es una CNN simple de clasificacion. Es un modelo de deteccion y segmentacion.

Si preguntan que hace `predict_dental_xray.py`:

> Ejecuta los modelos locales sobre una radiografia y genera imagenes rotuladas y reportes CSV.

Si preguntan que hace `dental_xray_pipeline.py`:

> Prepara y audita los datasets, genera archivos en formato YOLO y permite entrenar o reentrenar modelos.

Si preguntan que hace `model_registry.py`:

> Centraliza las rutas y metadatos de los modelos locales para que todo el proyecto use la misma fuente de verdad.
