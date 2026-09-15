"""Lead scoring — TFM Máster Big Data, Data Science e IA (UCM).

Productivización del modelo descrito en el documento final del TFM. Ofrece dos modos de uso:

1. Consulta individual: los datos de un cliente y su propensión.
2. Priorización de cartera: un fichero CSV con muchos clientes, devuelto ya ordenado
   por propensión y con el dimensionado económico de la campaña.

Decisiones del modelo que condicionan esta aplicación:

- El modelo se entrena sin reponderado de clases, de forma deliberada. Las tres
  estrategias probadas (sin balanceo, reponderado y SMOTENC) ordenan a los clientes
  prácticamente igual, pero solo la primera produce probabilidades calibradas, y la
  regla de decisión de esta aplicación depende de que lo estén.
- El modelo no utiliza el género del cliente ni su región. Ninguna de las dos aporta
  capacidad predictiva apreciable, y en el caso del género se trata además de una
  categoría protegida en el acceso a servicios financieros.

Todas las constantes proceden de `metadatos.json`, que genera el notebook.
"""

import io
import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Lead Scoring · Tarjeta de crédito",
                   page_icon="💳", layout="wide")


@st.cache_resource
def cargar_pipeline():
    """Carga el pipeline (preprocesamiento + modelo). Se cachea entre ejecuciones."""
    return joblib.load("modelo_pipeline.pkl")


@st.cache_data
def cargar_json(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


pipeline = cargar_pipeline()
opciones = cargar_json("opciones.json")
meta = cargar_json("metadatos.json")

TASA_BASE = meta["tasa_base"]
RANGOS = meta["rangos"]
COLUMNAS = meta["columnas_entrada"]

st.title("💳 Propensión a contratar una tarjeta de crédito")
st.caption(
    f"Modelo XGBoost entrenado con {meta['n_entrenamiento']:,} clientes reales. "
    f"ROC-AUC de {meta['roc_auc_test']} sobre un conjunto de test independiente de "
    f"{meta['n_test']:,} clientes que no intervino en ninguna decisión."
)

# --- Economía de la campaña -------------------------------------------------
# Para un modelo calibrado, contactar a un cliente compensa cuando el valor
# esperado supera el coste: p * VALOR > COSTE, es decir, p > COSTE / VALOR.
# Esta regla solo es válida porque las probabilidades del modelo son reales.
with st.sidebar:
    st.header("Economía de la campaña")
    st.caption("Ajusta estos valores a los de tu caso. El punto de corte se recalcula solo.")
    coste = st.number_input("Coste de contactar a un cliente (€)", 0.1, 1000.0, 5.0, 0.5)
    valor = st.number_input("Margen por cliente captado (€)", 0.1, 10000.0, 40.0, 5.0)

    umbral = min(coste / valor, 1.0)
    st.metric("Punto de corte resultante", f"{umbral:.1%}")
    st.caption(
        f"Contactar compensa cuando la probabilidad supera {umbral:.1%}, porque a partir "
        "de ahí el margen esperado cubre el coste del contacto."
    )

    st.divider()
    st.caption(
        "El modelo no utiliza el género ni la región del cliente: no aportan capacidad "
        "predictiva apreciable y el género es además una categoría protegida."
    )


def aviso_extrapolacion(edad, vintage):
    """El formulario permite combinaciones que apenas aparecen en los datos de
    entrenamiento, y ahí la predicción es poco fiable."""
    antiguedad_anios = vintage / 12
    return (edad - antiguedad_anios) < 21 or (edad < 30 and vintage > 48)


tab_individual, tab_cartera = st.tabs(["Un cliente", "Cartera completa"])

# =========================================================== CLIENTE INDIVIDUAL
with tab_individual:
    st.write(
        "Introduce los datos de un cliente y el modelo estima la probabilidad de que "
        "esté interesado en una tarjeta de crédito, y si compensa contactarlo."
    )

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Edad", RANGOS["Age"][0], RANGOS["Age"][1], 45)
        occupation = st.selectbox("Ocupación", opciones["Occupation"], index=3)
        channel_code = st.selectbox("Canal de captación", opciones["Channel_Code"], index=2)
        is_active = st.selectbox("¿Cliente activo?", opciones["Is_Active"], index=1)

    with col2:
        vintage = st.slider("Antigüedad con el banco (meses)",
                            RANGOS["Vintage"][0], RANGOS["Vintage"][1], 60)
        credit_product = st.selectbox(
            "¿Tiene producto de crédito?",
            opciones["Credit_Product"],
            help=(
                "'Unknown' son clientes sin ese dato registrado en el sistema. No es un "
                "valor perdido cualquiera: en los datos históricos ese grupo presenta una "
                "tasa de contratación del 85%, muy por encima del resto."
            ),
        )
        avg_balance = st.number_input(
            "Saldo medio en cuenta",
            min_value=RANGOS["Avg_Account_Balance"][0],
            max_value=RANGOS["Avg_Account_Balance"][1],
            value=900_000,
            step=10_000,
            help=(f"Rango observado en los datos: de "
                  f"{RANGOS['Avg_Account_Balance'][0]:,} a "
                  f"{RANGOS['Avg_Account_Balance'][1]:,}."),
        )

    st.divider()

    if st.button("Calcular propensión", type="primary"):
        cliente = pd.DataFrame([{
            "Occupation": occupation,
            "Channel_Code": channel_code,
            "Credit_Product": credit_product,
            "Is_Active": is_active,
            "Age": age,
            "Vintage": vintage,
            "Avg_Account_Balance": avg_balance,
        }])

        proba = float(pipeline.predict_proba(cliente)[0, 1])
        valor_esperado = proba * valor - coste

        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidad de contratación", f"{proba:.1%}")
        c2.metric("Frente al cliente medio", f"{proba / TASA_BASE:.1f}x")
        c3.metric("Margen esperado del contacto", f"{valor_esperado:,.1f} €")

        if proba >= umbral:
            st.success(
                f"Contactar. Con un punto de corte del {umbral:.1%}, este cliente supera "
                "el umbral y el contacto tiene un margen esperado positivo."
            )
        else:
            st.info(
                f"No priorizar. Por debajo del punto de corte del {umbral:.1%}, el coste "
                "del contacto no se compensa con el margen esperado."
            )

        if aviso_extrapolacion(age, vintage):
            st.warning(
                "Esta combinación de edad y antigüedad es muy poco frecuente en los datos "
                "de entrenamiento. El modelo está extrapolando y la estimación es menos "
                "fiable de lo habitual."
            )

        with st.expander("Ver los datos enviados al modelo"):
            st.dataframe(cliente, use_container_width=True)

# ============================================================ CARTERA COMPLETA
with tab_cartera:
    st.write(
        "Sube un fichero CSV con la cartera y la aplicación devuelve la lista ordenada "
        "por propensión, junto con el dimensionado económico de la campaña. Es el modo "
        "de uso que corresponde a una campaña real: el modelo no decide sobre un cliente, "
        "decide **a cuántos y a cuáles** contactar."
    )

    with st.expander("Formato esperado del fichero"):
        st.write("Una fila por cliente y, como mínimo, estas columnas:")
        st.code(", ".join(COLUMNAS))
        st.write(
            "Cualquier columna adicional (por ejemplo un identificador) se conserva en "
            "el resultado. Los valores de las variables categóricas deben ser los que "
            "aparecen en los desplegables de la otra pestaña."
        )
        plantilla = pd.DataFrame([{
            "ID": "CLIENTE_0001",
            "Occupation": opciones["Occupation"][0],
            "Channel_Code": opciones["Channel_Code"][0],
            "Credit_Product": opciones["Credit_Product"][0],
            "Is_Active": opciones["Is_Active"][0],
            "Age": 45, "Vintage": 60, "Avg_Account_Balance": 900_000,
        }])
        st.download_button("Descargar plantilla CSV",
                           plantilla.to_csv(index=False).encode("utf-8"),
                           "plantilla_cartera.csv", "text/csv")

    fichero = st.file_uploader("Fichero CSV de la cartera", type=["csv"])

    if fichero is not None:
        try:
            cartera = pd.read_csv(fichero)
        except Exception as e:
            st.error(f"No se ha podido leer el fichero: {e}")
            st.stop()

        faltan = [c for c in COLUMNAS if c not in cartera.columns]
        if faltan:
            st.error("Faltan columnas obligatorias: " + ", ".join(faltan))
            st.stop()

        # Se avisa de los valores categóricos que el modelo no ha visto nunca. El
        # codificador los ignora sin fallar, pero conviene que el usuario lo sepa.
        avisos = []
        for col, validos in opciones.items():
            desconocidos = set(cartera[col].dropna().unique()) - set(validos)
            if desconocidos:
                avisos.append(f"{col}: {', '.join(map(str, sorted(desconocidos)))}")
        if avisos:
            st.warning(
                "Hay valores que no aparecen en los datos de entrenamiento. Se puntúan "
                "igualmente, pero la estimación es menos fiable:\n\n- " + "\n- ".join(avisos)
            )

        with st.spinner(f"Puntuando {len(cartera):,} clientes..."):
            cartera["propension"] = pipeline.predict_proba(cartera[COLUMNAS])[:, 1]

        cartera["margen_esperado"] = cartera["propension"] * valor - coste
        cartera["contactar"] = cartera["propension"] >= umbral
        cartera = cartera.sort_values("propension", ascending=False).reset_index(drop=True)
        cartera.insert(0, "prioridad", cartera.index + 1)

        n_contactar = int(cartera["contactar"].sum())
        margen_total = float(cartera.loc[cartera["contactar"], "margen_esperado"].sum())
        captacion_esperada = float(cartera.loc[cartera["contactar"], "propension"].sum())

        st.subheader("Dimensionado de la campaña")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clientes a contactar", f"{n_contactar:,}",
                  f"{n_contactar / len(cartera):.1%} de la cartera")
        c2.metric("Captaciones esperadas", f"{captacion_esperada:,.0f}")
        c3.metric("Margen neto esperado", f"{margen_total:,.0f} €")
        c4.metric("Coste de la campaña", f"{n_contactar * coste:,.0f} €")

        st.caption(
            f"Con un coste de {coste:.2f} € por contacto y un margen de {valor:.2f} € por "
            f"captación, compensa contactar a todo cliente con propensión superior al "
            f"{umbral:.1%}. Las captaciones esperadas son la suma de las probabilidades, "
            "que es una estimación válida precisamente porque el modelo está calibrado."
        )

        st.subheader("Cartera priorizada")
        st.dataframe(
            cartera.head(200).style.format({"propension": "{:.1%}",
                                            "margen_esperado": "{:,.1f} €"}),
            use_container_width=True,
        )
        if len(cartera) > 200:
            st.caption(f"Se muestran los 200 primeros de {len(cartera):,}. "
                       "La descarga incluye la cartera completa.")

        buffer = io.StringIO()
        cartera.to_csv(buffer, index=False)
        st.download_button("Descargar cartera priorizada (CSV)",
                           buffer.getvalue().encode("utf-8"),
                           "cartera_priorizada.csv", "text/csv", type="primary")

# --- Pie ---------------------------------------------------------------------
st.divider()
st.caption(
    f"ROC-AUC de {meta['roc_auc_test']} y Brier score de {meta['brier_test']} sobre el "
    "conjunto de test."
)
