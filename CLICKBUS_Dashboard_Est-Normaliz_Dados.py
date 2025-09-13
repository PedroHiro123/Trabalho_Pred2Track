# Dashboard para obter estatísticas e definir a normalização de dados

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -------------------------
# Configurações
# -------------------------
DATA_PATH = "/content/drive/MyDrive/dados-CLICKBUS/df_t_MODIFICADO.csv"
LOGO_PATH = "/content/drive/MyDrive/dados-CLICKBUS/ClickBus_logo.png"

st.set_page_config(layout="wide", page_title="Dashboard Estatísticas Originais")

# -------------------------
# Título do Dashboard e Logotipo
# -------------------------

col_title, col_logo = st.columns([0.8, 0.2])

with col_title:
    st.title("📊 Estatística Inicial para Normalização dos Dados")
    
with col_logo:
    try:
        image = Image.open(LOGO_PATH)
        st.image(image, width=150)
    except FileNotFoundError:
        st.error("Erro: Arquivo do logo não encontrado. Verifique o caminho.")

# -------------------------
# Carregamento dos dados
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=True)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Erro ao carregar o arquivo: {e}")
    st.stop()

# -------------------------
# Estatísticas descritivas
# -------------------------
st.header("📋 Estatísticas Descritivas — Variáveis Numéricas")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if numeric_cols:
    desc = df[numeric_cols].describe().T
    desc["mediana"] = df[numeric_cols].median()
    desc["n_valores_unicos"] = df[numeric_cols].nunique()

    # Renomeando para Português (PT-BR)
    desc = desc.rename(columns={
        "count": "Contagem (n)",
        "mean": "Média",
        "std": "Desvio padrão",
        "min": "Mínimo",
        "25%": "1º Quartil (Q1)",
        "50%": "2º Quartil (Mediana/Q2)",
        "75%": "3º Quartil (Q3)",
        "max": "Máximo"
    })

    st.dataframe(desc)

    st.markdown("""
    **Explicação das estatísticas apresentadas:**
    - **Contagem (n):** quantidade de registros não nulos considerados no cálculo.
    - **Média:** soma de todos os valores dividida pela quantidade (tendência central).
    - **Desvio padrão:** medida de dispersão que indica quanto os valores variam em torno da média.
    - **Mínimo / Máximo:** menor e maior valor encontrado na coluna.
    - **Quartis (Q1, Q2, Q3):** valores que dividem os dados em 4 partes iguais.
        - Q1: 25% dos valores são menores ou iguais.
        - Q2: Mediana (50% dos valores abaixo e 50% acima).
        - Q3: 75% dos valores são menores ou iguais.
    - **Mediana:** valor central dos dados ordenados, mais robusto a outliers que a média.
    - **Valores únicos:** quantidade de valores distintos encontrados na coluna.
    """)

# -------------------------
# Missing values
# -------------------------
st.header("❓ Análise de Valores Ausentes (Missing)")

missing = df.isna().sum().rename("Quantidade de valores ausentes").to_frame()
missing["% de valores ausentes"] = (missing["Quantidade de valores ausentes"] / len(df) * 100).round(2)
st.dataframe(missing)

if missing["Quantidade de valores ausentes"].sum() == 0:
    st.success("✅ Não foram encontrados valores ausentes explícitos (NaN).")
else:
    st.warning("⚠️ Existem valores ausentes (NaN). Devem ser avaliados com o time de negócio antes de qualquer imputação.")

st.markdown("""
**O que foi considerado:** - Valores nulos explícitos (`NaN`) foram identificados e contabilizados.
- Valores "disfarçados" (como `0` ou `"NA"`) **não foram considerados** como missing aqui, pois podem ser legítimos para o negócio.
""")

# -------------------------
# Outliers (IQR)
# -------------------------
st.header("📈 Análise de Outliers (Método do IQR)")

def detect_outliers(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return 0
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))).sum())

outlier_summary = {col: detect_outliers(df[col]) for col in numeric_cols}
out_df = pd.DataFrame.from_dict(outlier_summary, orient="index", columns=["Qtde de outliers (IQR)"])
st.dataframe(out_df)

if out_df["Qtde de outliers (IQR)"].sum() == 0:
    st.success("✅ Nenhum outlier identificado pelas regras de IQR.")
else:
    st.info("ℹ️ Foram identificados outliers estatísticos. No entanto, no contexto de negócio (passagens/tickets), valores extremos podem ser legítimos e **não devem ser tratados automaticamente**.")

st.markdown("""
**O que foi considerado:** - **Outliers pelo método do IQR (Interquartile Range):** - Intervalo interquartílico (IQR) = Q3 - Q1
  - Valores abaixo de **Q1 - 1,5×IQR** ou acima de **Q3 + 1,5×IQR** são considerados outliers.
- Esse método é estatístico e **não avalia o contexto de negócio**.
""")

# -------------------------
# Categorias (Paretto)
# -------------------------
st.header("🏷 Análise de Categorias — Paretto (80/20)")

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
selected_col = st.selectbox("Selecione uma coluna categórica:", cat_cols)

def paretto_table(df, col):
    freq = df[col].value_counts(dropna=False).reset_index()
    freq.columns = [col, "Frequência"]
    freq["%"] = freq["Frequência"] / freq["Frequência"].sum() * 100
    freq["% acumulado"] = freq["%"].cumsum()
    return freq

if selected_col:
    freq = paretto_table(df, selected_col)
    st.dataframe(freq.head(20))

    fig, ax = plt.subplots()
    ax.bar(freq[selected_col].astype(str).head(20), freq["%"].head(20))
    ax.plot(freq[selected_col].astype(str).head(20),
            freq["% acumulado"].head(20), color="red", marker="o")
    ax.axhline(80, color="green", linestyle="--")
    plt.xticks(rotation=90)
    st.pyplot(fig)

    if (freq["% acumulado"] >= 80).idxmax() < len(freq) / 2:
        st.success(f"✅ A coluna `{selected_col}` segue a regra de Paretto: poucos valores concentram a maioria dos registros.")
    else:
        st.info(f"ℹ️ A coluna `{selected_col}` apresenta distribuição mais equilibrada entre categorias.")

st.markdown("""
**O que foi considerado:** - **Princípio de Paretto (80/20):** aproximadamente 80% dos registros costumam estar concentrados em 20% das categorias.
- Essa análise ajuda a identificar colunas dominadas por poucos valores frequentes.
""")

# -------------------------
# Conclusão final
# -------------------------
st.header("📝 Conclusão Geral")

conclusions = []
if missing["Quantidade de valores ausentes"].sum() == 0:
    conclusions.append("Não foram encontrados valores ausentes explícitos (NaN).")
else:
    conclusions.append("Foram encontrados valores ausentes (NaN). Devem ser avaliados com o time de negócio.")

if out_df["Qtde de outliers (IQR)"].sum() == 0:
    conclusions.append("Nenhum outlier estatístico identificado.")
else:
    conclusions.append("Foram identificados outliers estatísticos, mas no contexto de negócio podem ser legítimos.")

conclusions.append("Valores categóricos foram analisados via Paretto (80/20). Algumas colunas apresentam forte concentração, outras não.")

st.write("**Resumo:**")
for c in conclusions:
    st.write("- " + c)

st.info("👉 Conclusão: **Não realizar tratamento automático de missings/outliers. Avaliar cada caso com base no contexto do negócio e na relevância da variável para o modelo preditivo.**")