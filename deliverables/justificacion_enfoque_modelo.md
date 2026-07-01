# Justificacion del enfoque de modelado

Este texto esta pensado para centrar la presentacion en la decision metodologica principal del proyecto: **por que se eligio entrenar un modelo deep learning de clasificacion de piezas dentales a partir de recortes generados desde poligonos**.

## Idea central para presentar

El objetivo del proyecto no es clasificar una radiografia completa, sino reconocer piezas dentales individuales. Una radiografia panoramica contiene muchas piezas al mismo tiempo, por lo que entregar una unica etiqueta para toda la imagen no resuelve el problema. Por esa razon, el enfoque elegido consiste en usar las anotaciones por poligono del dataset para aislar cada diente, generar un recorte individual y entrenar una CNN que clasifique cada pieza.

En otras palabras, transformamos un problema complejo de radiografia completa en un problema supervisado mas controlado: cada entrada del modelo corresponde a una pieza dental y cada salida corresponde a su clase.

## Por que no clasificar la radiografia completa

Una alternativa simple habria sido entrenar una CNN usando la radiografia completa como entrada. Sin embargo, esa opcion no calza bien con el objetivo, porque una sola imagen puede contener 20, 25 o mas dientes visibles. Si se entrega una etiqueta unica a la imagen, se pierde la informacion de que pieza especifica se esta clasificando.

Ademas, en una panoramica dental hay estructuras repetidas y muy cercanas entre si. Los dientes comparten patrones visuales, pero cambian por posicion, forma, tamano y contexto anatomico. Por eso el modelo necesita observar cada pieza de forma individual, no toda la radiografia como una sola unidad.

La decision de trabajar con recortes permite que la CNN aprenda rasgos propios de cada pieza: forma de corona, tamano relativo, orientacion, proporcion raiz-corona y patrones morfologicos visibles en la radiografia.

## Por que usar los poligonos del dataset

Los poligonos son la parte mas valiosa del dataset porque indican la region exacta asociada a cada pieza dental. A diferencia de una caja rectangular, el poligono se ajusta mejor al contorno del diente y permite separar la pieza del fondo.

Esto nos conviene por tres razones:

1. Permite generar muestras individuales de entrenamiento.
2. Reduce ruido visual, porque el modelo se concentra en la pieza y no en toda la radiografia.
3. Usa la supervision mas fina disponible en el dataset.

Por eso, antes de entrenar la CNN, el pipeline toma cada poligono, genera una mascara, recorta la region dental y guarda ese recorte en la clase correspondiente.

## Por que usar una CNN

La CNN es adecuada porque las piezas dentales se diferencian por patrones espaciales y visuales. Las convoluciones permiten aprender bordes, texturas, curvas, cambios de intensidad y formas locales. Luego, al combinar varias capas, la red puede construir representaciones mas complejas de cada pieza.

El modelo implementado, `ToothPieceCNN`, usa bloques convolucionales 2D, normalizacion, activaciones ReLU, pooling y capas fully connected. Es un modelo deep learning supervisado, entrenado localmente con PyTorch, sin depender de servicios externos.

Esta decision tambien es coherente con la literatura revisada. El paper **"Tooth Segmentation in 3D CBCT Images Using Deep Convolutional Neural Network"** muestra que las CNN pueden aprender estructuras dentales desde imagen medica. El paper **"A Fully Automated Method for 3D Individual Tooth Identification and Segmentation in Dental CBCT"** refuerza que el problema relevante no es solo detectar dientes, sino identificar piezas individuales. Finalmente, **"Teeth Detection and Dental Problem Classification in Panoramic X-Ray Images Using Deep Learning and Image Processing"** apoya el uso de deep learning en radiografias panoramicas dentales.

Aunque esos trabajos no son identicos al nuestro, respaldan la idea principal: para analizar dientes en imagenes medicas conviene primero localizar o aislar estructuras dentales y luego clasificarlas.

## Por que esta estrategia es defendible

La estrategia es defendible porque conecta tres elementos:

- El dataset disponible trae poligonos por pieza dental.
- El objetivo del proyecto es clasificar piezas, no tratamientos ni radiografias completas.
- La literatura muestra que las redes convolucionales funcionan bien para segmentacion e identificacion dental.

Entonces, el flujo elegido aprovecha directamente la informacion del dataset. No se fuerza el problema a una clasificacion global, sino que se adapta el modelo a la forma real de las anotaciones.

## Flujo metodologico

El flujo aplicado es:

1. Cargar radiografias y anotaciones JSON.
2. Leer los objetos anotados como poligonos.
3. Extraer la clase de cada pieza.
4. Crear una mascara desde el poligono.
5. Recortar la pieza dental.
6. Redimensionar el recorte a un tamano fijo.
7. Dividir los datos en train, valid y test por radiografia.
8. Entrenar una CNN multiclase.
9. Evaluar con accuracy, macro precision, macro recall, macro F1-score y matriz de confusion.

La division por radiografia es importante porque evita que dientes de la misma imagen aparezcan simultaneamente en entrenamiento y validacion. Eso hace que la evaluacion sea mas honesta.

## Resultados de preparacion de datos

La preparacion completa del dataset produjo:

| Split | Radiografias | Recortes dentales |
|---|---:|---:|
| Train | 478 | 12198 |
| Valid | 59 | 1518 |
| Test | 61 | 1602 |
| Total | 598 | 15318 |

El modelo trabaja con 32 clases, correspondientes a las clases numericas del dataset.

## Como explicarlo oralmente

> Elegimos este enfoque porque el problema real no es clasificar una radiografia completa, sino clasificar piezas dentales individuales. Como el dataset ya trae poligonos por diente, usamos esa informacion para recortar cada pieza y convertirla en una muestra de entrenamiento. De esta forma, cada entrada de la CNN tiene una etiqueta clara y representa una sola pieza dental.
>
> Esta decision reduce ruido, aprovecha mejor las anotaciones disponibles y se alinea con la literatura, donde los trabajos mas relevantes primero segmentan o identifican estructuras dentales antes de clasificarlas. Por eso entrenamos una CNN propia sobre recortes de dientes, evaluandola con metricas de clasificacion como accuracy, precision, recall, F1-score y matriz de confusion.

## Mensaje final para la diapositiva

La decision metodologica principal fue **adaptar el entrenamiento a la forma del dataset**: como las etiquetas vienen por poligono y el objetivo es clasificar piezas individuales, se transformaron las radiografias en recortes dentales supervisados. Esto permite entrenar un modelo deep learning mas coherente con el problema, mas interpretable y mas facil de evaluar.
