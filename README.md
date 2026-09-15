# Lead Scoring de tarjetas de crédito — TFM

Trabajo Fin de Máster del Máster en Big Data, Data Science e Inteligencia Artificial
(Universidad Complutense de Madrid).
Modalidad elegida: **opción 1, análisis de un dataset con orientación Data Scientist**.

**Prototipo desplegado:** https://pruebatfm-5uu6rosmtmung2ccxoxfsw.streamlit.app

---

## El problema

Un banco quiere saber a qué clientes de su cartera merece la pena ofrecerles una tarjeta de crédito. Contactar a todo el mundo es caro y molesta a quien no está interesado; no contactar a nadie deja negocio sin estrategias que le permitan adquirir clientes por otros canales. El modelo ordena la cartera por probabilidad de contratación para que la campaña se dirija a quien más probablemente responda, y permite dimensionarla según el presupuesto disponible.

## Datos

- **Origen:** *Credit Card Lead Prediction*, procedente del JOB-A-THON de Analytics
  Vidhya (mayo de 2021), publicado en Kaggle. (https://www.kaggle.com/datasets/adityasharma95/credit-card-lead-prediction-analytics-vidhya)
- **Volumen:** 245.725 clientes, 9 variables explicativas.
- **Variable objetivo:** `Is_Lead`, con un 23,7 % de casos positivos.
- Los ficheros de datos no se incluyen en este repositorio; se descargan de Kaggle.
- Solo se utiliza `train_s3TEQDk.csv`. El fichero `test_mSzZ8RL.csv` no trae la variable
  objetivo, porque era el de envío de la competición original, así que el conjunto de
  test de este trabajo se obtiene particionando el primero.

## Resultados principales

| Métrica sobre el conjunto de test (49.145 clientes, nunca usados para decidir nada) | Valor |
|---|---|
| ROC-AUC | ≈ 0,873 |
| PR-AUC | ≈ 0,748 |
| Brier score | ≈ 0,104 |

Lectura en términos de campaña: **contactando al 20 % de la cartera se alcanza en torno
al 62 % de los clientes interesados**, algo más del triple de lo que se conseguiría
contactando al azar.

> Las cifras exactas las imprime el notebook al ejecutarse y quedan registradas en
> `metadatos.json`, que es lo que lee la aplicación.

## Contenido del repositorio

| Fichero | Descripción |
|---|---|
| `app.py` | Prototipo Streamlit. Dos modos: consulta de un cliente, y priorización de una cartera completa subida en CSV |
| `modelo_pipeline.pkl` | Pipeline completo (preprocesamiento + modelo XGBoost) entrenado con los 245.725 registros. Solo scikit-learn y XGBoost, sin transformadores propios |
| `opciones.json` | Valores válidos de cada variable categórica, usados para poblar el formulario |
| `metadatos.json` | Métricas, rangos de las variables y lista de columnas de entrada. La aplicación no tiene ninguna constante escrita a mano |
| `requirements.txt` | Versiones exactas de las dependencias |

## Cómo reproducir el trabajo

Todo el análisis está en un **único notebook**, `TFM_Lead_Scoring_DECP.ipynb`, que
se ejecuta de principio a fin en Google Colab en unos 15 minutos aproximadamente. 
Hay que subir previamente `train_s3TEQDk.csv` a la sesión.

El notebook recorre el análisis descriptivo, las transformaciones, la comparación de
técnicas con ajuste de hiperparámetros e intervalos de confianza por *bootstrap*, el
tratamiento del desbalance, el análisis de calibración, la selección del punto de corte a
partir de la economía de la campaña, el estudio de ablación, la revisión de equidad, la
evaluación final sobre test, la interpretabilidad con SHAP y la generación de los
artefactos de producción de este repositorio.

Todos los procesos aleatorios usan `random_state = 42`, de modo que los resultados son
reproducibles ejecución tras ejecución.

**Dependencias y versiones.** `scikit-learn` cambia entre versiones la forma de
serializar algunos objetos internos, así que el `.pkl` solo carga correctamente con las
versiones fijadas en `requirements.txt`. En macOS, `xgboost` necesita además la
biblioteca OpenMP, que se instala con `brew install libomp`.

## Decisiones metodológicas que conviene conocer

**Los valores ausentes de `Credit_Product` no se imputan.** Ese grupo, un 11,9 % de la
cartera, presenta una tasa de contratación del 85 %, frente al 31 % de quienes sí tienen
producto de crédito y el 7 % de quienes no. La ausencia del dato no es aleatoria y
resulta informativa por sí misma, así que se codifica como una categoría más
(`Unknown`). El estudio de ablación cuantifica la decisión: eliminar esa variable cuesta
cerca de diez puntos de ROC-AUC, e imputarla equivaldría a perder buena parte de esa
información.

**El modelo final no repondera las clases.** Se compararon tres estrategias: sin
balanceo, reponderado y sobremuestreo con SMOTENC. Las tres ordenan a los clientes
prácticamente igual, pero solo la primera produce probabilidades calibradas, y la regla
de decisión del prototipo depende de ello. El modelo final no utiliza ningún dato
sintético.

**El punto de corte no está fijado en 0,5.** Para un modelo calibrado, contactar
compensa cuando la probabilidad supera el cociente entre el coste del contacto y el
margen por cliente captado. El barrido empírico de umbrales confirma ese valor teórico,
lo que funciona además como verificación independiente de la calibración. El prototipo
permite ajustar ambos importes y recalcula el umbral en consecuencia.

**El modelo no utiliza el género ni la región del cliente.** El estudio de ablación
muestra que ninguna de las dos aporta capacidad predictiva apreciable. En el caso del
género, además, se trata de una categoría protegida en el acceso a servicios
financieros, de modo que utilizarla solo añadiría riesgo regulatorio sin contrapartida.
El notebook comprueba también que el modelo ordena igual de bien a hombres y a mujeres.
Retirar `Region_Code` tiene un beneficio técnico adicional: el pipeline pasa a ser
scikit-learn puro y desaparece la dependencia de un transformador propio, que era el
punto más frágil del despliegue.

## Limitación principal

El poder predictivo del modelo descansa en buena medida sobre `Credit_Product`: esa sola
variable recupera en torno al 85 % de la capacidad discriminante del modelo completo,
medida sobre el exceso de ROC-AUC respecto al azar. Y su valor más informativo es
precisamente el dato ausente, que parece reflejar en qué punto del proceso comercial se
encuentra el cliente más que una característica suya. Si el banco modifica la forma de
registrar ese campo, el modelo perderá buena parte de su capacidad y habrá que
reentrenarlo. El notebook incluye un plan de monitorización orientado a detectar
precisamente esa deriva.
