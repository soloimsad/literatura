# Sintesis de Articulos

Esta sintesis se construyo desde los textos extraidos en `docs/literature_review/full_texts/`. La prioridad practica es apoyar la revision bibliografica y la comparacion entre metodos.

## 1. Conditional Diffusion Models for CT Image Synthesis from CBCT

Archivo extraido: `1-Conditional_Diffusion_Models_for_CT_Image_Synthesis_from.txt`

Tema: revision sistematica sobre modelos de difusion condicional para generar CT sintetico desde CBCT en radioterapia.

Aporte principal:

- Resume estrategias de difusion condicional para traduccion CBCT -> CT.
- Enfatiza reduccion de artefactos, fidelidad anatomica, precision de HU y posible utilidad en planificacion/dosis.
- Compara de forma cualitativa con enfoques previos como GANs y VAEs.

Metricas/reportes relevantes:

- MAE en HU, PSNR, SSIM.
- Gamma passing rate y desviaciones dosimetricas cuando estan disponibles.
- Se mencionan mejoras como 35% en MAE y 30% en PSNR para modelos de difusion texture-preserving frente a GANs en estudios revisados.

Limitaciones importantes:

- Alta heterogeneidad entre datasets, anatomias, dimensionalidad, supervision y metricas.
- No se puede concluir superioridad uniforme con una unica comparacion agregada.
- Se necesitan protocolos estandarizados y validacion multiinstitucional.

Relevancia para el proyecto:

- Alta si el proyecto trata reconstruccion/restauracion/sintesis de imagen medica.
- Media si el foco es segmentacion dental, porque aporta buenas ideas de evaluacion cuantitativa y validacion clinica, pero no es dental directo.

## 2. Tooth Segmentation in 3D CBCT Images Using Deep Convolutional Neural Network

Archivo extraido: `2-Tooth_Segmentation_in_3D_CBCT_Using_CNN.txt`

Tema: segmentacion binaria de dientes en volumenes 3D CBCT usando una CNN 3D.

Aporte principal:

- Usa 70 volumenes 3D CBCT de una instalacion clinica local.
- Genera ground truth con un metodo semiautomatico.
- Entrena una red CNN 3D de 38 capas con extraccion de parches y data augmentation.
- Usa Generalized Dice Loss para manejar el fuerte desbalance de clases entre diente y fondo.

Metricas/resultados:

- Validacion reportada: aproximadamente 95.54% a 95.57% de accuracy.
- Comparacion de modelos:
  - 3D U-Net: mean IoU 0.60, mean Dice 0.86, accuracy 95.75%, entrenamiento 62 h, tamano 70 MB.
  - Modelo propuesto/DRNet: mean IoU 0.70, mean Dice 0.90, accuracy 95.54%, entrenamiento 23 h, tamano 4.3 MB.
  - 3D ResNet: mean IoU 0.48, mean Dice 0.36, accuracy 80%, entrenamiento 18 h, tamano 121.4 MB.

Limitaciones:

- Dataset pequeno y local.
- Ground truth semiautomatico.
- Segmenta dientes vs fondo, no necesariamente instancias individuales de cada pieza dental.
- Algunas regiones de mandibula pueden ser clasificadas erroneamente como dientes.

Relevancia para el proyecto:

- Muy alta si el trabajo sera sobre segmentacion de dientes en CBCT.
- Buen candidato como baseline o paper central de la revision.

## 3. A Fully Automated Method for 3D Individual Tooth Identification and Segmentation in Dental CBCT

Archivo extraido: `3-A_Fully_Automated_Method_For_3D_Individual_Tooth.txt`

Tema: identificacion y segmentacion automatica de dientes individuales en CBCT dental.

Aporte principal:

- Convierte el CBCT 3D a imagenes panoramicas 2D para reducir complejidad.
- Detecta e identifica dientes en 2D.
- Proyecta ROIs loose/tight hacia el volumen 3D.
- Segmenta dientes individuales con una FCN 3D tipo U-shaped.

Metricas/resultados:

- Identificacion dental: F1-score 93.35%.
- Segmentacion 3D individual: Dice similarity coefficient 94.79%.
- Para segmentacion 3D con loose + tight ROIs: precision 95.97%, recall 93.71%, DSC 94.79%, Hausdorff Distance 1.66 mm, ASSD 0.14 mm.
- En deteccion de bounding boxes, AP 88.11% con threshold IoU 0.6.

Limitaciones:

- Pipeline mas complejo que una red end-to-end simple.
- Depende de la calidad de la proyeccion panoramica y de los ROIs.
- Las comparaciones directas pueden depender mucho del dataset y del protocolo.

Relevancia para el proyecto:

- Muy alta para un proyecto de CBCT dental, especialmente si se quiere segmentacion por instancia o identificacion de piezas.
- Excelente referencia para justificar metricas: F1, Dice, HD, ASSD, precision y recall.

## 4. Teeth Detection and Dental Problem Classification in Panoramic X-Ray Images Using Deep Learning and Image Processing

Archivo extraido: `4-Teeth_Detection_and_Dental_Problem_Classification.txt`

Tema: deteccion de dientes y clasificacion de problemas dentales en radiografias panoramicas.

Aporte principal:

- Combina segmentacion semantica, generacion/refinamiento de bounding boxes y clasificacion por votacion mayoritaria dentro de la region detectada.
- Usa imagenes panoramicas obtenidas desde tres clinicas dentales.
- Considera problemas como restauraciones, implantes, dentaduras y otros.

Metricas/resultados:

- Evaluado sobre mas de 200 imagenes ground truth.
- Solucion propuesta: accuracy 0.89, precision 0.98, recall 0.91, F1-score 0.93.
- Metodo 1: accuracy 0.87, precision 0.98, recall 0.89, F1-score 0.93.
- Metodo 2: accuracy 0.68, precision 0.98, recall 0.68, F1-score 0.80.
- Tiempo promedio de ejecucion reportado para la solucion propuesta: 60 s.

Limitaciones:

- El tiempo de ejecucion es alto.
- El paper indica como trabajo futuro mejorar runtime e incluir mas clases semanticas.
- Es panoramica 2D, no CBCT 3D.

Relevancia para el proyecto:

- Alta si el proyecto usa radiografias panoramicas o clasificacion de problemas dentales.
- Media si el proyecto es CBCT 3D, porque aun sirve para deteccion/clasificacion y metricas.

## 5. Image Captioning with CNN and LSTM using Python

Archivo extraido: `5-Image_Captioning_with_CNN_and_LSTM_using_Python.txt`

Tema: generacion de captions de imagenes usando CNN como encoder visual y RNN/LSTM como decoder de lenguaje.

Aporte principal:

- Explica la arquitectura general CNN + LSTM/RNN para image captioning.
- Discute enfoques como Neural Image Caption, atencion top-down/bottom-up y uso de datasets como Flickr8k.
- Incluye una aplicacion con backend/frontend y almacenamiento de captions.

Metricas/resultados:

- La evaluacion es principalmente cualitativa por casos de prueba.
- Reporta captions exitosos para imagenes generales, pero falla cuando requiere identificar una persona especifica.
- Concluye "moderate accuracy" usando Flickr8k, sin una tabla clara de BLEU, METEOR, ROUGE o CIDEr en el texto extraido.

Limitaciones:

- Menor relacion con imagenes medicas.
- Validacion debil si se compara con papers que reportan metricas cuantitativas estandar.
- No es una fuente fuerte para justificar evaluacion medica o dental.

Relevancia para el proyecto:

- Baja para segmentacion/clasificacion medica.
- Media solo si el proyecto incluye descripcion automatica de imagenes o generacion de texto desde imagenes.

## 6. A Python Programmed ResNet-50 CNN Designed for Low Light Image Enhancement

Archivo extraido: `6-A_Python_Programmed_Resnet-50_CNN_Designed_for_Low_Light_Image_Enhancement.txt`

Tema: mejora de imagenes de baja iluminacion usando una CNN basada en ResNet-50.

Aporte principal:

- Propone una arquitectura ResNet-50 para mejorar brillo, claridad y contraste.
- Menciona mapas multiescala y SSIM loss para preservar texturas.
- Implementacion reportada en Python.

Metricas/resultados:

- Menciona PSNR y SSIM como metricas de evaluacion.
- El texto extraido no muestra una tabla cuantitativa robusta con valores de PSNR/SSIM.
- Afirma mejor desempeno frente a tecnicas tradicionales, pero la evidencia cuantitativa visible es limitada.

Limitaciones:

- No es imagen medica ni dental de forma directa.
- Validacion cuantitativa insuficiente en el texto extraido.
- Puede servir mas como referencia de preprocessing que como paper central.

Relevancia para el proyecto:

- Media si se requiere mejorar contraste/calidad visual antes de segmentar o clasificar.
- Baja si el trabajo no involucra restauracion o enhancement.

## 7. Performance Analysis of Convolutional Neural Network Models

Archivo extraido: `7-Performance_Analysis_of_Convolutional_Neural_Network_Models.txt`

Tema: analisis de rendimiento de AlexNet, VGG-16 y ResNet-50 en distintas plataformas, incluyendo GPU, CPU, Jetson TX2 y FPGA.

Aporte principal:

- Evalua inferencia y entrenamiento de CNNs populares.
- Compara plataformas de ejecucion.
- Propone una arquitectura de convolucion para FPGA Virtex-7.

Metricas/resultados:

- Dataset de evaluacion: ImageNet, input 224 x 224.
- Throughput FPGA:
  - AlexNet: 184.32 GFLOPS.
  - VGG-16: 172.8 GFLOPS.
  - ResNet-50: 197.09 GFLOPS.

Limitaciones:

- No evalua metricas clinicas ni calidad de segmentacion/clasificacion medica.
- Es una fuente de rendimiento computacional, no de metodologia dental.

Relevancia para el proyecto:

- Media si el proyecto necesita justificar eficiencia, hardware, tiempo de entrenamiento o eleccion de arquitectura.
- Baja como referencia principal del estado del arte medico.

## Recomendacion de Uso en la Revision

Prioridad alta para un proyecto dental/CBCT:

- Paper 2: segmentacion de dientes en 3D CBCT con CNN.
- Paper 3: identificacion y segmentacion individual 3D en CBCT.
- Paper 4: deteccion/clasificacion dental en panoramicas, si el proyecto incluye radiografias 2D o clasificacion.

Prioridad media:

- Paper 1: fuerte para reconstruccion/sintesis y discusion de metricas/validacion.
- Paper 6: util si se plantea preprocessing/enhancement.
- Paper 7: util si se discute costo computacional o ejecucion en hardware.

Prioridad baja:

- Paper 5, salvo que el proyecto incluya captioning, generacion textual o interpretabilidad narrativa de imagenes.
