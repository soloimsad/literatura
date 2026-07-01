# Resumen de articulos alineado a la rubrica INF-402

Este resumen esta organizado segun lo que pide la pauta de evaluacion de INF-402 para la revision bibliografica: no basta con decir de que trata cada articulo, tambien conviene identificar objetivo, metodologia, resultados, metricas, fortalezas, limitaciones y relevancia para el proyecto.

La rubrica premia especialmente:

- seleccion de articulos relevantes e indexados;
- analisis critico, no solo resumen;
- sintesis comparativa entre metodos;
- justificacion de la metodologia propuesta;
- claridad para explicar resultados y limitaciones.

En este proyecto, el foco practico esta en radiografias dentales, segmentacion/clasificacion de piezas y deteccion de posibles hallazgos o tratamientos usando modelos de vision por computador. Por eso, los articulos mas cercanos al proyecto son los que tratan segmentacion dental, identificacion de piezas y clasificacion de problemas dentales.

## Criterios de evaluacion usados para construir el resumen

La pauta de `docs/evaluation/` evalua dos niveles. Primero, la presentacion de revision bibliografica, donde el mayor puntaje esta en seleccionar buenos articulos, analizarlos criticamente y compararlos. Segundo, el proyecto final, donde se exige que la metodologia, los resultados, las metricas y el codigo sean reproducibles. Por eso este resumen no esta escrito como una lista neutral de papers, sino como material para defender decisiones del proyecto.

| Criterio de la pauta | Puntaje | Como se aborda en este resumen |
|---|---:|---|
| Seleccion y calidad de articulos | 20 | Se separan articulos centrales, de apoyo y menos alineados, priorizando los que realmente se conectan con radiografias dentales, segmentacion, identificacion de piezas y clasificacion de hallazgos. |
| Analisis critico | 25 | Cada articulo incluye fortalezas, limitaciones y relevancia para el proyecto, para evitar una presentacion puramente descriptiva. |
| Sintesis y comparacion | 20 | Se incluye una tabla comparativa y una lectura transversal de brechas, tendencias y oportunidades. |
| Claridad y estructura | 15 | Los articulos estan organizados por utilidad para el proyecto, no solo por orden numerico. Esto ayuda a explicar mejor en 15 a 20 minutos. |
| Dominio oral | 10 | Se destacan argumentos defendibles para preguntas: por que usar CNN/YOLO, por que separar modelos, que metricas usar y que limitaciones reconocer. |
| Referencias IEEE | 5 | Este resumen no reemplaza las referencias formales; para la entrega final se deben escribir las citas completas en formato IEEE. |

Para el proyecto final, la misma pauta pide que los resultados se expliquen con metricas adecuadas segun la tarea. En este caso hay dos tareas principales: deteccion/segmentacion de piezas dentales y deteccion/clasificacion de tratamientos o hallazgos. Por eso, las metricas que conviene mencionar son:

| Tarea del proyecto | Metricas mas defendibles segun la pauta | Que significan en palabras simples |
|---|---|---|
| Segmentacion de piezas o regiones dentales | Dice Score, IoU, Hausdorff Distance | Miden que tan bien coincide la region predicha con la region real. Dice e IoU evaluan superposicion; Hausdorff evalua error de borde o distancia. |
| Clasificacion de pieza dental o tratamiento | Accuracy, Precision, Recall, F1-score, AUC-ROC | Miden si el modelo asigna la clase correcta. Precision ayuda a discutir falsos positivos; recall ayuda a discutir falsos negativos; F1 resume ambos. |
| Preprocesamiento, mejora o reconstruccion de imagen | PSNR, SSIM, RMSE | Solo aplican si se defiende una etapa de mejora de imagen. No son las metricas centrales del sistema actual, pero sirven para papers de apoyo. |

La lectura estrategica es esta: los articulos 2, 3 y 4 son los que mas ayudan a subir puntaje en seleccion, analisis critico y justificacion metodologica. Los articulos 1, 6 y 7 sirven como soporte para hablar de calidad de imagen, preprocesamiento y eficiencia computacional. El articulo 5 debe tratarse con cuidado porque aporta una idea futura, pero no justifica directamente la tarea dental principal.

## Vista comparativa rapida

| # | Articulo | Tipo de imagen/tarea | Modelo/metodo | Metricas reportadas | Relevancia para el proyecto |
|---|---|---|---|---|---|
| 1 | Conditional Diffusion Models for CT Image Synthesis from CBCT | Sintesis/reconstruccion CBCT -> CT | Modelos de difusion condicional, comparados con GAN/CNN/VAE | MAE, PSNR, SSIM, metricas dosimetricas | Media: util para discutir calidad de imagen y validacion, pero no es dental directo |
| 2 | Tooth Segmentation in 3D CBCT Using CNN | Segmentacion dental 3D | CNN 3D / DRNet, comparado con 3D U-Net y 3D ResNet | Accuracy, Dice, IoU | Alta: fundamenta segmentacion dental con CNN |
| 3 | Fully Automated 3D Individual Tooth Identification and Segmentation | Identificacion y segmentacion individual en CBCT | Pipeline 2D + 3D FCN tipo U-shaped | F1, Dice, Precision, Recall, Hausdorff, ASSD | Muy alta: referencia fuerte para piezas individuales y metricas |
| 4 | Teeth Detection and Dental Problem Classification in Panoramic X-Ray | Deteccion dental y clasificacion de problemas en panoramicas | CNN + segmentacion semantica + bounding boxes + clasificacion | Accuracy, Precision, Recall, F1 | Muy alta: es el mas cercano al proyecto actual en panoramicas 2D |
| 5 | Image Captioning with CNN and LSTM | Captioning de imagenes generales | CNN encoder + LSTM decoder | Evaluacion cualitativa, moderate accuracy | Baja: sirve solo como referencia general de CNN + secuencias |
| 6 | ResNet-50 CNN for Low Light Image Enhancement | Mejoramiento de imagenes de baja iluminacion | ResNet-50 CNN | PSNR, SSIM | Media-baja: util si se habla de preprocesamiento/enhancement |
| 7 | Performance Analysis of CNN Models | Rendimiento computacional de CNNs | AlexNet, VGG-16, ResNet-50 en GPU/CPU/FPGA | GFLOPS, throughput, tiempos | Media-baja: util para discutir costo computacional, no clinico |

## Lectura por aporte a la rubrica

Si se piensa en la nota, no todos los papers deben ocupar el mismo espacio. La pauta entrega mas puntos por analisis critico y comparacion que por cantidad de texto, entonces conviene usar cada articulo con una funcion clara:

| Funcion dentro de la presentacion | Articulos recomendados | Justificacion |
|---|---|---|
| Paper principal para conectar con el sistema implementado | Articulo 4 | Trabaja con radiografias panoramicas, deteccion de dientes y clasificacion de problemas dentales. Es el paralelo mas directo con los modelos locales del proyecto. |
| Soporte fuerte para segmentacion e identificacion dental | Articulos 2 y 3 | Demuestran que las CNN son adecuadas para segmentar estructuras dentales y que identificar piezas individuales es una tarea relevante. |
| Soporte metodologico para metricas de imagen medica | Articulo 1 | Ayuda a explicar que en imagen medica se debe validar con metricas cuantitativas y no solo con inspeccion visual. |
| Soporte para preprocesamiento | Articulo 6 | Sirve si se discute calidad de imagen, contraste o enhancement antes de la inferencia. |
| Soporte para ejecucion local y rendimiento | Articulo 7 | Ayuda a responder por que importa el hardware, el tiempo de inferencia y el costo computacional de las CNN. |
| Extension futura, no base metodologica | Articulo 5 | Puede usarse para hablar de generacion automatica de reportes, pero no debe defender el nucleo del proyecto. |

La forma mas solida de exponer es agruparlos asi: primero el problema dental directo, luego las arquitecturas de segmentacion/clasificacion, despues las metricas, y al final las limitaciones o extensiones. Eso responde mejor a la rubrica que presentar siete resumenes aislados.

## Articulo 1 - Conditional Diffusion Models for CT Image Synthesis from CBCT

**Tema general.**  
Este articulo trata sobre modelos de difusion condicional para generar imagenes CT sinteticas a partir de CBCT. Su foco esta en radioterapia y reconstruccion/sintesis de imagen medica, no especificamente en odontologia. La motivacion principal es que las imagenes CBCT suelen tener artefactos, ruido y menor precision de unidades Hounsfield en comparacion con CT convencional. Por eso, transformar CBCT en un CT sintetico de mayor calidad puede ser util para planificacion clinica y analisis cuantitativo.

**Objetivo del articulo.**  
El objetivo es revisar y comparar enfoques recientes de difusion condicional aplicados a la generacion de CT sintetico desde CBCT. La pregunta central no es detectar objetos, sino mejorar o sintetizar una modalidad de imagen a partir de otra.

**Metodologia.**  
El articulo funciona como revision sistematica. Resume distintos trabajos que usan modelos de difusion condicional y los compara con enfoques previos como GANs, VAEs, CNNs y modelos basados en transformers. Se enfoca en calidad de imagen, reduccion de artefactos, preservacion anatomica y utilidad dosimetrica.

**Tipo de red o enfoque.**  
Los modelos de difusion no son exactamente CNN clasicas de clasificacion. Son modelos generativos que aprenden un proceso de eliminacion gradual de ruido para producir una imagen final. En muchos casos incorporan arquitecturas convolucionales o U-Net internas, pero el paradigma principal es generativo.

**Metricas relevantes.**  
Las metricas mencionadas son tipicas de reconstruccion o restauracion de imagen:

- MAE en unidades Hounsfield;
- PSNR;
- SSIM;
- errores dosimetricos;
- gamma passing rate cuando se evalua planificacion radioterapeutica.

Estas metricas son distintas a las que se usan en segmentacion dental, donde normalmente se reportan Dice, IoU o Hausdorff Distance.

**Fortalezas.**  
El articulo es fuerte para mostrar como se evalua calidad de imagen medica. Tambien sirve para demostrar que en imagenes medicas no basta con que una imagen "se vea bien": se requieren metricas cuantitativas y validacion clinica. Es una buena referencia para hablar de reconstruccion, restauracion y calidad cuantitativa.

**Limitaciones.**  
No esta directamente alineado con el proyecto si el foco es deteccion de piezas dentales y tratamientos en radiografias panoramicas. Ademas, al ser una revision de multiples estudios, las comparaciones pueden ser dificiles porque los datasets, anatomias, protocolos y metricas no siempre son equivalentes.

**Relevancia para el proyecto.**  
Su relevancia es media. No es un paper central para justificar YOLO ni segmentacion dental, pero puede servir para la introduccion o discusion si se quiere hablar de calidad de imagen, necesidad de validacion cuantitativa y desafios generales en imagen medica.

**Como usarlo en la presentacion.**  
Conviene mencionarlo brevemente como contexto amplio: la imagen medica moderna no solo clasifica o segmenta, tambien reconstruye, mejora y sintetiza imagenes. Sin embargo, no deberia ocupar mucho tiempo en la presentacion porque no esta directamente conectado con radiografia dental panoramica.

## Articulo 2 - Tooth Segmentation in 3D CBCT Images Using Deep Convolutional Neural Network

**Tema general.**  
Este articulo aborda la segmentacion de dientes en volumenes CBCT 3D mediante redes neuronales convolucionales profundas. A diferencia de una radiografia panoramica 2D, CBCT entrega un volumen tridimensional, lo que permite analizar la anatomia dental con mayor detalle espacial.

**Objetivo del articulo.**  
El objetivo es segmentar automaticamente dientes dentro de imagenes CBCT. La tarea principal es separar regiones dentales del fondo o de estructuras no dentales. Esto es relevante porque segmentar dientes manualmente en volumenes 3D puede ser lento, costoso y dependiente del operador.

**Metodologia.**  
El trabajo usa un conjunto de volumenes CBCT y genera ground truth mediante un procedimiento semiautomatico. Luego entrena una CNN 3D con extraccion de parches y aumento de datos. La red aprende a clasificar voxeles o regiones como diente/fondo, enfrentando el desbalance natural entre estructuras dentales y fondo.

**Tipo de red o enfoque.**  
El enfoque es claramente una CNN 3D. Esto significa que las convoluciones operan sobre volumenes, no solo sobre imagenes 2D. El articulo compara el modelo propuesto con arquitecturas como 3D U-Net y 3D ResNet.

**Metricas relevantes.**  
Las metricas reportadas son muy utiles para la rubrica porque se alinean con segmentacion:

- Accuracy;
- mean IoU;
- mean Dice;
- tiempo de entrenamiento;
- tamano del modelo.

El articulo reporta resultados cercanos a 95.5% de accuracy y valores de Dice/IoU comparables con baselines.

**Fortalezas.**  
Es una referencia fuerte porque trabaja directamente con dientes y usa CNN para segmentacion. Ademas, compara con otros modelos, lo que ayuda a construir analisis critico. Tambien aporta una idea importante: no solo importa la metrica, tambien el costo computacional y el tamano del modelo.

**Limitaciones.**  
El dataset es relativamente pequeno y local. El ground truth no es completamente manual, sino semiautomatico, lo que puede introducir sesgos. Ademas, el problema se enfoca mas en segmentacion binaria de dientes versus fondo, no necesariamente en identificar cada pieza individual ni en clasificar tratamientos.

**Relevancia para el proyecto.**  
La relevancia es alta si se quiere justificar el uso de CNNs o modelos deep learning para segmentacion dental. Aunque el proyecto actual use radiografias panoramicas 2D y YOLO-seg, este articulo muestra que las CNN son efectivas para extraer estructuras dentales en imagen medica.

**Como usarlo en la presentacion.**  
Este articulo se puede usar para explicar que la segmentacion dental automatica ya es una linea activa de investigacion. Sirve para justificar metricas como Dice e IoU y para explicar que las redes convolucionales son adecuadas para reconocer estructuras anatomicas dentales.

## Articulo 3 - A Fully Automated Method for 3D Individual Tooth Identification and Segmentation in Dental CBCT

**Tema general.**  
Este articulo propone un metodo automatizado para identificar y segmentar dientes individuales en CBCT dental 3D. A diferencia del articulo anterior, no se queda solo en separar dientes del fondo, sino que busca reconocer piezas individuales.

**Objetivo del articulo.**  
El objetivo es resolver dos tareas: identificacion de dientes y segmentacion individual. Esto es mas cercano a un sistema clinico util, porque en odontologia no basta con decir "hay dientes"; interesa saber que pieza es cada una y donde esta.

**Metodologia.**  
El metodo combina varias etapas. Primero transforma el volumen CBCT 3D en una representacion panoramica 2D para reducir la complejidad. Luego detecta e identifica dientes en esa proyeccion. Despues proyecta regiones de interes hacia el volumen 3D y aplica una red de segmentacion 3D tipo U-shaped para obtener cada diente individual.

**Tipo de red o enfoque.**  
Es un pipeline hibrido, no una sola red simple de extremo a extremo. Combina procesamiento 2D, deteccion de regiones de interes y segmentacion 3D con una FCN de forma U. Este enfoque muestra que en problemas dentales complejos puede ser util dividir el problema en etapas.

**Metricas relevantes.**  
Este articulo reporta metricas muy alineadas con la pauta:

- F1-score para identificacion dental;
- Dice Similarity Coefficient para segmentacion;
- Precision;
- Recall;
- Hausdorff Distance;
- ASSD;
- Average Precision para bounding boxes.

Reporta F1 alrededor de 93.35% para identificacion y Dice cercano a 94.79% para segmentacion individual.

**Fortalezas.**  
Es uno de los articulos mas fuertes para la revision porque combina identificacion y segmentacion individual, que son tareas muy cercanas al proyecto. Tambien usa metricas robustas y variadas, lo que ayuda a comparar desde distintas perspectivas: exactitud de clase, calidad de segmentacion y distancia geometrica.

**Limitaciones.**  
El pipeline es mas complejo que un modelo unico como YOLO. Depende de la calidad de la proyeccion panoramica, de las regiones de interes y de la segmentacion posterior. Esto puede hacerlo mas dificil de implementar y mantener. Ademas, al trabajar con CBCT 3D, no es directamente equivalente a radiografias panoramicas 2D.

**Relevancia para el proyecto.**  
La relevancia es muy alta como referencia conceptual. Aunque el proyecto actual use YOLO-seg en imagenes 2D, comparte la idea de detectar y clasificar piezas individuales. Tambien es muy util para justificar por que se reportan clases, confianza, ubicacion y potencialmente mascaras.

**Como usarlo en la presentacion.**  
Conviene usarlo como ejemplo de estado del arte avanzado. Puede servir para decir que los sistemas mas completos no solo segmentan dientes, sino que tambien identifican piezas individualmente. Eso conecta bien con el modelo `tooth` del proyecto.

## Articulo 4 - Teeth Detection and Dental Problem Classification in Panoramic X-Ray Images Using Deep Learning and Image Processing

**Tema general.**  
Este articulo trata sobre deteccion de dientes y clasificacion de problemas dentales en radiografias panoramicas. Es el articulo mas cercano al proyecto actual porque trabaja con imagenes 2D panoramicas, no con CBCT 3D.

**Objetivo del articulo.**  
El objetivo es detectar estructuras dentales y clasificar problemas o condiciones visibles en radiografias panoramicas. Esto se parece mucho al flujo del proyecto, donde se usan modelos para detectar piezas dentales y posibles hallazgos/tratamientos.

**Metodologia.**  
El articulo combina deep learning e image processing. Usa una CNN para segmentacion semantica, luego genera o refina bounding boxes y finalmente clasifica problemas dentales dentro de las regiones detectadas. El enfoque no depende solo de una red, sino de un pipeline que combina prediccion de la red y procesamiento posterior.

**Tipo de red o enfoque.**  
El enfoque se basa en CNN para extraer patrones visuales en radiografias panoramicas. Es distinto a una CNN de clasificacion global, porque trabaja con regiones y deteccion de objetos/problemas. Conceptualmente se acerca a lo que hace YOLO: localizar objetos y asignar clases.

**Metricas relevantes.**  
Reporta metricas de clasificacion/deteccion muy alineadas con la pauta:

- Accuracy;
- Precision;
- Recall;
- F1-score.

El articulo reporta, para su solucion propuesta, accuracy cercana a 0.89, precision 0.98, recall 0.91 y F1-score 0.93. Estas metricas son muy faciles de explicar en la presentacion porque conectan con rendimiento de clasificacion.

**Fortalezas.**  
Es altamente relevante porque trabaja sobre panoramicas dentales, igual que el flujo practico del proyecto. Tambien considera problemas dentales y no solo piezas, por lo que conecta bien con el modelo `treatment`. Ademas, reporta metricas claras y comparables.

**Limitaciones.**  
El tiempo de ejecucion reportado es relativamente alto, cercano a 60 segundos en la solucion propuesta. El articulo tambien menciona como trabajo futuro mejorar runtime e incluir mas clases semanticas. Otra limitacion es que depende de una secuencia de procesamiento, por lo que errores en una etapa pueden afectar las siguientes.

**Relevancia para el proyecto.**  
La relevancia es muy alta. Este articulo puede ser el centro de la justificacion del proyecto porque comparte modalidad de imagen, tipo de tarea y uso de deep learning. Sirve para comparar con el enfoque actual basado en YOLO-seg.

**Como usarlo en la presentacion.**  
Conviene presentarlo como el paper mas cercano al sistema implementado. Puede usarse para justificar que detectar piezas y clasificar problemas en panoramicas es un problema real, evaluable y relevante. Tambien sirve para comparar metricas de accuracy, precision, recall y F1.

## Articulo 5 - Image Captioning with CNN and LSTM using Python

**Tema general.**  
Este articulo trata sobre generacion automatica de descripciones de imagenes usando CNN y LSTM. La tarea se conoce como image captioning: el modelo recibe una imagen y produce una frase que describe su contenido.

**Objetivo del articulo.**  
El objetivo es construir un sistema que combine vision por computador y procesamiento de lenguaje natural. La CNN actua como extractor visual, mientras que la LSTM genera la secuencia de palabras.

**Metodologia.**  
El enfoque usa una CNN como encoder de caracteristicas visuales y una red recurrente LSTM como decoder de texto. Tambien discute datasets generales de imagenes, como Flickr8k, y presenta una implementacion tipo aplicacion.

**Tipo de red o enfoque.**  
Es un modelo hibrido CNN + LSTM. La CNN procesa la imagen, pero la LSTM se encarga de modelar la secuencia de palabras. No es un modelo de deteccion ni segmentacion.

**Metricas relevantes.**  
En el texto extraido, la evaluacion parece principalmente cualitativa. No se observa una tabla robusta con metricas tipicas de captioning como BLEU, METEOR, ROUGE o CIDEr. Esto limita su utilidad para una revision bibliografica fuerte.

**Fortalezas.**  
Sirve para explicar que las CNN pueden usarse como extractores de caracteristicas visuales en sistemas mas amplios. Tambien muestra una integracion entre vision y lenguaje, que podria ser interesante si el proyecto futuro quisiera generar reportes automaticos.

**Limitaciones.**  
No esta directamente relacionado con odontologia ni imagen medica. Tampoco se centra en segmentacion, deteccion de objetos ni clasificacion dental. Su evaluacion cuantitativa es debil en comparacion con los articulos dentales.

**Relevancia para el proyecto.**  
La relevancia es baja. Puede mencionarse solo como referencia secundaria si se quiere hablar de posibles extensiones, por ejemplo generar descripciones automaticas de hallazgos. No deberia ser un paper central si el tiempo de presentacion es limitado.

**Como usarlo en la presentacion.**  
Si se incluye, conviene hacerlo muy brevemente. Puede servir para decir que una posible extension futura seria generar reportes automaticos desde las detecciones, pero no es la base metodologica del proyecto actual.

## Articulo 6 - A Python Programmed ResNet-50 CNN Designed for Low Light Image Enhancement

**Tema general.**  
Este articulo aborda el mejoramiento de imagenes de baja iluminacion usando una red ResNet-50. Aunque no es un articulo dental ni medico directo, esta relacionado con preprocesamiento y mejora de calidad visual.

**Objetivo del articulo.**  
El objetivo es mejorar brillo, claridad y contraste de imagenes afectadas por baja iluminacion. El supuesto es que una imagen de mejor calidad visual puede facilitar tareas posteriores de analisis o clasificacion.

**Metodologia.**  
El articulo propone una implementacion en Python basada en ResNet-50. ResNet-50 es una CNN profunda con conexiones residuales, disenada para permitir entrenar redes mas profundas sin degradar tanto el aprendizaje.

**Tipo de red o enfoque.**  
El enfoque se basa en una CNN profunda, especificamente ResNet-50. Las conexiones residuales permiten que la red aprenda transformaciones complejas y preserve informacion relevante. En este caso, la red se orienta a enhancement, no a deteccion.

**Metricas relevantes.**  
Las metricas asociadas a este tipo de tarea son:

- PSNR;
- SSIM.

Estas metricas son propias de restauracion o mejoramiento de imagen, y coinciden con la pauta del curso para tareas de restauracion/reconstruccion. Sin embargo, en el texto extraido no se observa una tabla cuantitativa tan robusta como en otros articulos.

**Fortalezas.**  
Sirve para hablar de preprocesamiento. En imagenes medicas, mejorar contraste o reducir ruido puede ser relevante antes de aplicar segmentacion o clasificacion. Tambien permite introducir ResNet como arquitectura CNN importante.

**Limitaciones.**  
No esta directamente enfocado en radiografias dentales. Tampoco evalua deteccion de piezas ni problemas odontologicos. La evidencia cuantitativa visible en el texto extraido parece limitada.

**Relevancia para el proyecto.**  
La relevancia es media-baja. Puede ser util si se quiere justificar una etapa de preprocesamiento o discutir que la calidad de imagen afecta el rendimiento de los modelos. Pero no deberia desplazar a los articulos dentales centrales.

**Como usarlo en la presentacion.**  
Puede mencionarse en una diapositiva de "trabajos relacionados secundarios" o "posibles mejoras futuras". Por ejemplo: una mejora futura podria incluir enhancement previo de radiografias antes de la inferencia.

## Articulo 7 - Performance Analysis of Convolutional Neural Network Models

**Tema general.**  
Este articulo analiza el rendimiento computacional de modelos CNN populares como AlexNet, VGG-16 y ResNet-50 en distintas plataformas, incluyendo CPU, GPU, Jetson TX2 y FPGA.

**Objetivo del articulo.**  
El objetivo no es resolver un problema clinico, sino comparar rendimiento de arquitecturas CNN y plataformas de ejecucion. Se enfoca en costo computacional, throughput e implementacion eficiente.

**Metodologia.**  
Evalua modelos conocidos sobre ImageNet y compara desempeno en distintas plataformas. Tambien propone una arquitectura de convolucion para FPGA. El analisis se centra en eficiencia de ejecucion mas que en calidad clinica.

**Tipo de red o enfoque.**  
El articulo trabaja con CNNs clasicas y conocidas:

- AlexNet;
- VGG-16;
- ResNet-50.

Estas arquitecturas son relevantes historicamente porque muestran la evolucion de modelos convolucionales para vision por computador.

**Metricas relevantes.**  
Las metricas principales son de rendimiento computacional:

- GFLOPS;
- throughput;
- tiempo de ejecucion;
- comparacion entre hardware.

No reporta metricas clinicas como Dice, IoU, F1 o AUC para imagen medica.

**Fortalezas.**  
Es util para discutir eficiencia y costo computacional. En un proyecto que debe ejecutarse en un equipo local, esto puede ser relevante para justificar decisiones practicas como tamano de imagen, uso de CPU/GPU o eleccion de modelos mas eficientes.

**Limitaciones.**  
No es un articulo dental ni medico. No sirve para justificar directamente el desempeno clinico del sistema. Tampoco evalua segmentacion dental ni clasificacion de tratamientos.

**Relevancia para el proyecto.**  
La relevancia es media-baja. Sirve como apoyo si se quiere hablar de eficiencia de modelos CNN, pero no como paper central de la revision bibliografica.

**Como usarlo en la presentacion.**  
Puede usarse para responder preguntas sobre rendimiento o para explicar que las CNN tienen costos computacionales importantes. No conviene dedicarle mucho tiempo si la presentacion esta ajustada.

## Sintesis critica segun la rubrica

Los articulos no tienen todos el mismo peso para el proyecto. Si se considera la rubrica, conviene priorizar aquellos que sean mas relevantes, tengan metricas claras y permitan justificar la metodologia.

**Articulos centrales recomendados:**

- Articulo 4, porque trabaja con radiografias panoramicas y clasificacion de problemas dentales.
- Articulo 3, porque aborda identificacion y segmentacion individual de dientes.
- Articulo 2, porque fundamenta segmentacion dental con CNN en imagen medica 3D.

**Articulos de apoyo:**

- Articulo 1, para discutir calidad de imagen medica, validacion cuantitativa y reconstruccion.
- Articulo 6, para hablar de preprocesamiento o enhancement.
- Articulo 7, para hablar de eficiencia computacional de CNNs.

**Articulo menos alineado:**

- Articulo 5, porque trata image captioning general y no deteccion/segmentacion dental.

## Brechas detectadas

Al comparar los articulos, aparecen varias brechas utiles para justificar el proyecto:

1. Muchos trabajos dentales fuertes se enfocan en CBCT 3D, pero el proyecto trabaja con radiografias panoramicas 2D. Esto justifica adaptar enfoques de deteccion/segmentacion a una modalidad mas comun y practica.
2. Algunos metodos son pipelines complejos con varias etapas. El uso de YOLO-seg puede simplificar la inferencia al combinar deteccion, clasificacion y segmentacion en un flujo mas directo.
3. Varios articulos reportan buenas metricas, pero no siempre entregan una demo simple y reproducible. El proyecto puede diferenciarse mostrando ejecucion local, imagenes rotuladas y reportes CSV.
4. Los articulos sobre tratamientos/hallazgos en panoramicas son especialmente relevantes, pero aun existe espacio para ampliar clases, mejorar runtime y combinar pieza dental con hallazgo clinico.
5. En imagen medica, las metricas deben elegirse segun la tarea: Dice/IoU/Hausdorff para segmentacion, Accuracy/Precision/Recall/F1/AUC para clasificacion, PSNR/SSIM/RMSE para restauracion.

## Justificacion metodologica para el proyecto

La literatura revisada permite justificar el uso de modelos deep learning basados en CNN para imagen dental. Los articulos 2 y 3 muestran que las CNN pueden segmentar estructuras dentales en CBCT, mientras que el articulo 4 muestra que redes profundas tambien pueden detectar dientes y clasificar problemas en radiografias panoramicas.

El proyecto usa modelos YOLO-seg, que son arquitecturas de deteccion y segmentacion basadas en redes convolucionales. Esta eleccion es coherente con la literatura porque:

- permite detectar multiples objetos en una misma radiografia;
- entrega ubicacion espacial de cada prediccion;
- puede generar mascaras o regiones segmentadas;
- produce clases y confianza por cada deteccion;
- se adapta bien a una demo visual para presentacion.

Ademas, separar el flujo en dos modelos, uno para pieza dental y otro para hallazgos/tratamientos, se justifica porque son tareas relacionadas pero distintas. Identificar una pieza dental no es lo mismo que detectar una caries, una corona o un implante. Esta separacion permite explicar mejor el objetivo de cada componente y mostrar resultados por modulo.

## Como llevar esto a diapositivas

Para una presentacion de revision bibliografica, conviene organizar estos articulos de la siguiente forma:

1. Una diapositiva de contexto: imagen dental, importancia clinica y desafios de segmentacion/clasificacion.
2. Una diapositiva con los tres papers centrales: articulos 2, 3 y 4.
3. Una tabla comparativa con modalidad, tarea, modelo, metricas, resultados y limitaciones.
4. Una diapositiva de brechas: complejidad de pipelines, diferencia CBCT vs panoramica, necesidad de demo reproducible.
5. Una diapositiva de justificacion: por que usar YOLO-seg y modelos locales separados por tarea.

Para el proyecto final, estos resumenes tambien sirven para escribir:

- Introduccion: problema clinico y tecnico.
- Estado del arte: comparacion de enfoques.
- Materiales y metodos: relacion entre tareas, modelos y metricas.
- Discusion: limitaciones y trabajo futuro.
