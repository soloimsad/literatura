# Metricas y Rubricas de Evaluacion - INF-402

Fuente principal: `0-Lineamientos_PRIM_INF402.pdf`

Este documento es la referencia de evaluacion del trabajo. Las dos evaluaciones cubiertas son:

- Presentacion de revision bibliografica: 15% de la nota final.
- Proyecto final: 45% de la nota final, dividido en manuscrito tipo paper 30% y presentacion oral 15%.

## Presentacion de Revision Bibliografica

Peso: 15% de la nota final.

Requisitos base:

- Minimo 7 articulos cientificos relevantes para el proyecto final.
- Deben venir de revistas o conferencias indexadas, por ejemplo IEEE, Springer, Elsevier, MICCAI o SPIE Medical Imaging.
- Se recomiendan publicaciones de los ultimos 5 anos indicados por el documento: 2021-2025.
- No se aceptan blogs, Wikipedia, enciclopedias online ni tesis como fuente primaria.
- Preprints de arXiv solo si tambien fueron publicados formalmente.
- Duracion: 15 a 20 minutos de presentacion mas 5 minutos de preguntas.
- Idioma: espanol, manteniendo terminos tecnicos en ingles cuando corresponda.
- Entrega del archivo: al menos 24 horas antes de la clase.
- Nombre sugerido: `RevBib_Apellido1_Apellido2_INF402.pptx` o `.pdf`.

Estructura minima:

1. Portada: titulo del proyecto, integrantes, fecha.
2. Introduccion y motivacion: problema clinico/tecnico, relevancia, pregunta o hipotesis.
3. Metodologia de busqueda: bases de datos, palabras clave, criterios de inclusion/exclusion.
4. Resumen de articulos: referencia completa, objetivo, metodologia, resultados, metricas, fortalezas, limitaciones y relevancia para el proyecto.
5. Analisis comparativo: tabla o figura comparativa, brechas, tendencias y justificacion metodologica.
6. Conclusiones: hallazgos principales y relacion con el diseno del proyecto final.
7. Referencias: formato IEEE.

Rubrica:

| Criterio | Puntaje |
|---|---:|
| Seleccion y calidad de articulos: 7+ articulos relevantes, indexados y recientes | 20 |
| Analisis critico: no solo resumen, tambien fortalezas, limitaciones y relevancia | 25 |
| Sintesis y comparacion: tabla/figura comparativa, tendencias, brechas y justificacion | 20 |
| Claridad y estructura: orden, visualidad y respeto del tiempo | 15 |
| Dominio oral: respuestas tecnicas y fluidez | 10 |
| Formato de referencias IEEE | 5 |
| Participacion en presentaciones de otros grupos | 5 |
| Total | 100 |

Prioridad practica para maximizar puntaje:

- No quedarse en resumen articulo por articulo: la rubrica premia mucho el analisis critico y la comparacion.
- Incluir una tabla comparativa con dataset, tarea, arquitectura, metricas, resultados, limitaciones y aporte al proyecto.
- Explicitar para cada paper por que sirve o no sirve para el proyecto final.
- Preparar preguntas/respuestas tecnicas, porque hay puntos asociados a dominio oral y participacion.

## Proyecto Final

Peso total: 45% de la nota final.

Componentes:

- Manuscrito IEEE TMI: 30% de la nota final.
- Presentacion oral: 15% de la nota final.

### Manuscrito IEEE TMI

Requisitos:

- Formato IEEE Transactions on Medical Imaging.
- Maximo 5 paginas, sin contar referencias ni apendices opcionales.
- Redaccion en espanol o ingles; el documento recomienda ingles.
- Entrega en PDF.
- Codigo fuente en `.zip`.
- Times New Roman 10 pt, doble columna segun template IEEE.
- Minimo 10 referencias en formato IEEE.

Estructura obligatoria:

1. Titulo, autores y abstract.
2. Introduccion.
3. Materiales y metodos.
4. Experimentos y resultados.
5. Discusion.
6. Conclusiones.
7. Referencias.
8. Apendices opcionales.

Metricas tecnicas esperadas segun tarea:

| Tipo de tarea | Metricas sugeridas por el documento |
|---|---|
| Segmentacion | Dice Score, IoU, Hausdorff Distance |
| Clasificacion | Accuracy, Precision, Recall, F1-score, AUC-ROC |
| Restauracion o reconstruccion | PSNR, SSIM, RMSE |

Rubrica del manuscrito:

| Criterio | Puntaje indicado |
|---|---:|
| Calidad del abstract e introduccion | 15 |
| Materiales y metodos: metodologia completa, replicable y justificada | 20 |
| Resultados y analisis: metricas, figuras, baselines y analisis riguroso | 25 |
| Discusion y conclusiones: interpretacion critica, limitaciones y trabajo futuro | 10 |
| Formato IEEE y referencias correctas | 10 |
| Calidad del codigo: limpio, comentado, reproducible e incluye README | 10 |
| Originalidad e impacto | 15 |
| Total declarado en el PDF | 100 |

Nota importante: los puntajes escritos en la tabla del PDF suman 105, aunque el documento declara "TOTAL 100 pts". Conviene confirmarlo con el docente si se va a usar esta tabla para planificar la distribucion exacta del esfuerzo.

### Codigo Fuente

Requisitos:

- Debe ser reproducible: otro evaluador con el mismo entorno debe poder ejecutarlo y obtener los mismos resultados.
- Debe incluir `README.md` con instrucciones claras de instalacion y ejecucion.
- Lenguajes aceptados: Python o MATLAB.
- Codigo comentado en las secciones principales.
- Si se usan bases de datos externas, incluir instrucciones o scripts de descarga.

Estructura recomendada:

```text
src/       Codigo fuente principal
data/      Datos de prueba o instrucciones para descargarlos
models/    Modelos entrenados o scripts de entrenamiento
results/   Figuras y tablas generadas
README.md  Instrucciones de uso
```

### Presentacion Oral del Proyecto Final

Requisitos:

- Duracion: 20 a 25 minutos mas 10 minutos de preguntas.
- Debe incluir demostracion en vivo o video del sistema/algoritmo.
- Archivo enviado al docente al menos 24 horas antes.

Estructura recomendada:

1. Portada.
2. Introduccion: problema, motivacion e hipotesis.
3. Estado del arte.
4. Materiales y metodos.
5. Resultados: figuras, tablas y metricas.
6. Discusion y conclusiones.
7. Demostracion.
8. Referencias principales.

Rubrica:

| Criterio | Puntaje |
|---|---:|
| Claridad y estructura | 15 |
| Dominio del contenido | 25 |
| Presentacion de resultados | 20 |
| Demostracion practica | 20 |
| Manejo del tiempo | 10 |
| Calidad de la comunicacion | 5 |
| Participacion en presentaciones de otros grupos | 5 |
| Total | 100 |

## Integridad Academica

- Todo codigo y texto debe ser propio o debidamente referenciado.
- Bases de datos publicas deben citarse con referencia y licencia.
- El plagio puede llevar a calificacion 0.
- El uso de IA generativa, por ejemplo ChatGPT o Copilot, debe declararse explicitamente en el manuscrito, indicando en que partes se uso y como.

## Entregas y Nombres de Archivo

| Entregable | Indicacion |
|---|---|
| Presentacion de revision bibliografica | Segun calendario; archivo 24 horas antes |
| Manuscrito IEEE TMI + codigo `.zip` | Enviar 48 horas antes de la presentacion oral |
| Presentacion oral del proyecto final | Segun calendario; archivo 24 horas antes |

Nombres:

- Manuscrito: `Paper_Apellido1_Apellido2_INF402.pdf`
- Codigo: `Codigo_Apellido1_Apellido2_INF402.zip`
- Presentacion final: `Presentacion_Apellido1_Apellido2_INF402.pdf`
