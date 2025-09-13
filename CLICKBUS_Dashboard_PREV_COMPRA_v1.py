import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configurações da Página ---
st.set_page_config(layout="wide", page_title="Análise e Previsão de Viagens")

# --- Funções Auxiliares para Análise e Visualização ---

def convert_df_to_csv(df):
    """Converte um DataFrame para CSV para download."""
    return df.to_csv(index=False).encode('utf-8')

def calculate_exponential_probabilities(avg_time):
    """
    Calcula a probabilidade de uma compra ocorrer em 7 e 30 dias usando
    a distribuição exponencial.
    """
    if pd.isna(avg_time) or avg_time <= 0:
        return np.nan, np.nan
    
    # Lambda (taxa de ocorrência) = 1 / tempo_medio
    rate_lambda = 1.0 / avg_time
    
    # Probabilidade (CDF) = 1 - e^(-lambda * x)
    prob_7d = 1 - np.exp(-rate_lambda * 7)
    prob_30d = 1 - np.exp(-rate_lambda * 30)
    
    return prob_7d * 100, prob_30d * 100

def plot_confusion_matrix_for_top_classes(y_true, y_pred, le_classes, top_n=10):
    """
    Plota a matriz de confusão para as N classes mais comuns no conjunto de dados,
    usando os rótulos originais.
    """
    # Identifica as classes mais comuns em y_true
    top_classes_codes = pd.Series(y_true).value_counts().nlargest(top_n).index
    top_classes_names = [le_classes[i] for i in top_classes_codes]

    # Filtra as previsões e os valores reais para incluir apenas as top classes
    mask = np.isin(y_true, top_classes_codes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Cria a matriz de confusão
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_classes_codes)

    # CORREÇÃO: Cria a figura e os eixos com o tamanho correto
    fig, ax = plt.subplots(figsize=(6, 4))  # Reduz o tamanho da figura
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=top_classes_names, yticklabels=top_classes_names, ax=ax)
    plt.title('Matriz de Confusão (Top 10 Classes)', fontsize=12)  # Diminui o tamanho da fonte do título
    plt.xlabel('Previsão', fontsize=10)
    plt.ylabel('Real', fontsize=10)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    st.pyplot(fig)

def plot_classification_metrics_bar_chart(report_dict, top_n=10):
    """
    Plota gráficos de barras para as métricas de precisão, recall e F1-score
    para as N classes mais comuns.
    """
    # Converte o dicionário do relatório para um DataFrame
    df_report = pd.DataFrame(report_dict).transpose().round(2)
    
    # Filtra as classes para remover a média e o suporte e pega as top N
    df_metrics = df_report.drop(columns=['support']).drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    
    # Ordena pelo F1-score e pega as top N classes
    top_metrics = df_metrics.nlargest(top_n, 'f1-score')

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # --- AJUSTE: Usa a paleta de cores 'coolwarm' ---
    colors = sns.color_palette('coolwarm')
    
    # Plota o gráfico de barras com as novas cores
    top_metrics[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax, width=0.8, color=colors)
    
    plt.title(f'Precisão, Recall e F1-Score para as {top_n} Classes Mais Relevantes')
    plt.xlabel('Classes (Destinos de Origem)')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)


# --- 1. Carregamento e Pré-processamento dos Dados ---
@st.cache_data
def load_and_preprocess_data(file_path):
    """
    Carrega e pré-processa os dados, treinando o modelo de ML de forma otimizada.
    """
    try:
        # Carrega apenas as colunas necessárias do CSV para economizar RAM
        cols_to_use = [
            'fk_contact',
            'date_purchase',
            'place_origin_departure',
            'place_destination_departure'
        ]
        df = pd.read_csv(file_path, usecols=cols_to_use)
        df['fk_contact'] = df['fk_contact'].astype(str)
        df['date_purchase'] = pd.to_datetime(df['date_purchase'])

        # Remove viagens com origem e destino iguais
        df = df[df['place_origin_departure'] != df['place_destination_departure']].copy()

        # Parte 1: Análise Estatística (Tempo entre compras)
        df.sort_values(by=['fk_contact', 'date_purchase'], inplace=True)
        df['time_diff'] = df.groupby('fk_contact')['date_purchase'].diff().dt.days

        df_agg = df.groupby('fk_contact')['time_diff'].mean().reset_index()
        df_agg.rename(columns={'time_diff': 'avg_time_between_purchases'}, inplace=True)
        df_agg.dropna(subset=['avg_time_between_purchases'], inplace=True)

        ultima_compra = df.groupby('fk_contact')['date_purchase'].max().reset_index()
        df_agg = pd.merge(df_agg, ultima_compra, on='fk_contact')
        
        # Parte 2: Treinamento do Modelo de Machine Learning
        # Filtra o DataFrame para incluir apenas clientes com mais de uma compra
        df_filtered = df[df['fk_contact'].isin(df_agg['fk_contact'])].copy()
        
        # Agrega o histórico de trechos por cliente
        df_history = df_filtered.groupby('fk_contact')['place_origin_departure'].apply(lambda x: ' '.join(x)).reset_index(name='history')
        
        # Cria a variável de destino (a última origem comprada)
        df_last_origin = df_filtered.sort_values('date_purchase').groupby('fk_contact')['place_origin_departure'].last().reset_index(name='last_origin')
        
        # Combina o histórico e a última origem
        df_ml = pd.merge(df_history, df_last_origin, on='fk_contact')

        # Converte a variável de destino para números
        origin_le = LabelEncoder()
        df_ml['last_origin_encoded'] = origin_le.fit_transform(df_ml['last_origin'])

        X = df_ml['history']
        y = df_ml['last_origin_encoded']
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(token_pattern=r'[^\s]+')),
            ('model', MultinomialNB())
        ])

        # Divide os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Garante que as classes do teste estão no conjunto de treino
        train_classes = np.unique(y_train)
        test_mask = np.isin(y_test, train_classes)
        X_test_filtered = X_test[test_mask]
        y_test_filtered = y_test[test_mask]
        
        if len(y_test_filtered) == 0:
            st.warning("Não há classes em comum entre os conjuntos de treino e teste. Não foi possível avaliar o modelo.")
            return df_agg, df, pipeline, origin_le, 0, {}, None, None, df_ml, None

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test_filtered)

        # Obtém as classes únicas presentes no conjunto de teste filtrado.
        unique_classes_test = np.unique(y_test_filtered)
        filtered_target_names = origin_le.classes_[unique_classes_test]
        
        ml_accuracy = accuracy_score(y_test_filtered, y_pred)
        
        ml_report = classification_report(y_test_filtered, y_pred, target_names=filtered_target_names, output_dict=True, zero_division=0)
            
        # Pré-calcula o destino mais comum para cada cliente e origem
        most_common_destinations = df.groupby(['fk_contact', 'place_origin_departure'])['place_destination_departure'].agg(pd.Series.mode)
        if isinstance(most_common_destinations, pd.Series):
            most_common_destinations = most_common_destinations.apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
        
        # Retorna todas as informações necessárias
        return df_agg, df, pipeline, origin_le, ml_accuracy, ml_report, y_test_filtered, y_pred, df_ml, most_common_destinations

    except FileNotFoundError:
        st.error(f"Erro: O arquivo de dados '{file_path}' não foi encontrado.")
        return None, None, None, None, None, None, None, None, None, None

def predict_trecho_vectorized(pipeline, origin_le, df_ml, most_common_destinations):
    """
    Retorna o trecho completo mais provável previsto pelo modelo de ML para todos os clientes,
    usando uma abordagem vetorizada para alta performance.
    """
    try:
        X_all = df_ml['history']
        predictions_encoded = pipeline.predict(X_all)
        predicted_origins = origin_le.inverse_transform(predictions_encoded)

        df_temp = pd.DataFrame({
            'fk_contact': df_ml['fk_contact'],
            'predicted_origin': predicted_origins
        })

        most_common_df = most_common_destinations.reset_index()
        most_common_df.rename(columns={
            'place_origin_departure': 'predicted_origin',
            most_common_df.columns[-1]: 'predicted_destination'
        }, inplace=True)
        
        df_predictions = pd.merge(df_temp, most_common_df, on=['fk_contact', 'predicted_origin'], how='left')
        df_predictions['predicted_destination'] = df_predictions['predicted_destination'].fillna('Destino desconhecido')
        df_predictions['trecho_previsto_ml'] = df_predictions['predicted_origin'] + ' -> ' + df_predictions['predicted_destination']
        
        return df_predictions[['fk_contact', 'trecho_previsto_ml']]
    
    except Exception as e:
        return pd.DataFrame({'fk_contact': [], 'trecho_previsto_ml': []})

def get_ml_prediction_single(customer_id, df_predictions):
    """
    Retorna a previsão do trecho para um único cliente a partir da tabela pré-calculada.
    """
    try:
        prediction_row = df_predictions[df_predictions['fk_contact'] == customer_id]
        if not prediction_row.empty:
            return prediction_row['trecho_previsto_ml'].iloc[0]
        else:
            return "Cliente não encontrado na base de treinamento do modelo."
    except Exception:
        return "Erro na previsão para este cliente."

# --- Execução Principal do Streamlit ---
file_path = '/content/drive/MyDrive/dados-CLICKBUS/df_t_MODIFICADO.csv'
logo_path = '/content/drive/MyDrive/dados-CLICKBUS/ClickBus_logo.png'

# Recebe os dados e métricas do pré-processamento e avaliação
df_agg, df_full, pipeline, origin_le, ml_accuracy, ml_report, y_test_filtered, y_pred, df_ml, most_common_destinations = load_and_preprocess_data(file_path)

if df_agg is not None and not df_agg.empty:
    
    total_clientes = df_full['fk_contact'].nunique()
    total_clientes_analisados = len(df_agg)

    df_predictions = predict_trecho_vectorized(pipeline, origin_le, df_ml, most_common_destinations)

    # Sidebar para Seleção de Cliente
    st.sidebar.header("Selecione o Cliente")
    lista_clientes = ['Todos'] + sorted(df_full['fk_contact'].unique().tolist())
    selected_customer_id = st.sidebar.selectbox("Selecione o ID do Cliente", lista_clientes)
    text_customer_id = st.sidebar.text_input("Ou digite o ID do Cliente:")
    
    customer_id = selected_customer_id
    if text_customer_id:
        customer_id = text_customer_id

    filtro_atual = "Todos os Clientes" if not customer_id or customer_id == 'Todos' else f"Cliente {customer_id}"
    st.sidebar.markdown(f'<p style="color:blue;">Filtro atual: <b>{filtro_atual}</b></p>', unsafe_allow_html=True)
    
    page_title = f"Dashboard de Previsão de Viagens: {filtro_atual}"
    
    col_title, col_logo = st.columns([4, 1])
    with col_title:
        st.title(page_title)
    with col_logo:
        try:
            st.image(logo_path, width=150)
        except FileNotFoundError:
            st.warning("A imagem do logo não foi encontrada. Verifique o caminho.")
    
    st.markdown("---")

    st.markdown(f"**Total de clientes na base:** `{total_clientes}`")
    st.markdown(f"**Total de clientes com histórico de compra:** `{total_clientes_analisados}`")
    
    # Seção de Previsão por Cliente
    st.subheader("Previsão da Próxima Compra")
    
    if customer_id and customer_id != 'Todos':
        df_single_client = df_full[df_full['fk_contact'] == customer_id].copy()
        
        if not df_single_client.empty:
            st.write(f"**ID do Cliente:** {customer_id}")
            
            customer_data_agg = df_agg[df_agg['fk_contact'] == customer_id]
            if not customer_data_agg.empty:
                avg_time = int(customer_data_agg['avg_time_between_purchases'].iloc[0])
                last_purchase_date = customer_data_agg['date_purchase'].iloc[0]
                proxima_data = last_purchase_date + timedelta(days=avg_time)
                avg_time_str = f"{avg_time}"
                data_prevista_str = proxima_data.strftime('%d/%m/%Y')
                
                prob_7d, prob_30d = calculate_exponential_probabilities(avg_time)
            else:
                avg_time_str = "N/A"
                data_prevista_str = "N/A"
                prob_7d, prob_30d = np.nan, np.nan

            predicted_trecho = get_ml_prediction_single(customer_id, df_predictions)
            
            st.markdown("---")
            # --- Ajuste do tamanho da fonte usando HTML/CSS ---
            st.markdown("### Previsão Completa da Próxima Compra")
            
            st.markdown(f"""
            <h5 style='font-weight: bold;'>Previsão da data e trecho da próxima compra:</h5>
            <div style='font-size: 18px; line-height: 1.6;'>
            <p><strong>Prob Est 7d(%):</strong> {prob_7d:.2f}%</p>
            <p><strong>Prob Est 30d(%):</strong> {prob_30d:.2f}%</p>
            <p><strong>Prev Est Próx Compra(dias):</strong> {avg_time_str}</p>
            <p><strong>Data Prevista:</strong> {data_prevista_str}</p>
            <p><strong>Trecho Previsto:</strong> {predicted_trecho}</p>
            </div>
            """, unsafe_allow_html=True)
            # --- Fim do ajuste da fonte ---

            df_export_single = pd.DataFrame({
                'ID Cliente': [customer_id],
                'Prob Est 7d(%)': [f"{prob_7d:.2f}%" if pd.notna(prob_7d) else 'N/A'],
                'Prob Est 30d(%)': [f"{prob_30d:.2f}%" if pd.notna(prob_30d) else 'N/A'],
                'Prev Est Próx Compra(dias)': [avg_time_str],
                'Data Prevista': [data_prevista_str],
                'Trecho Previsto (ML)': [predicted_trecho]
            })
            st.download_button(
                label="📥 Baixar Previsão do Cliente",
                data=convert_df_to_csv(df_export_single),
                file_name=f'previsao_completa_{customer_id}.csv',
                mime='text/csv'
            )

            # --- Histórico de Compras do Cliente ---
            st.markdown("---")
            st.markdown("### Histórico de Compras")
            df_history_client = df_single_client.copy()
            
            # Garante que as datas estão em ordem decrescente
            df_history_client.sort_values(by='date_purchase', ascending=False, inplace=True)

            df_history_client['Data Compra'] = df_history_client['date_purchase'].dt.strftime('%d/%m/%Y')
            df_history_client['Trecho (Origem -> Destino)'] = df_history_client['place_origin_departure'] + ' -> ' + df_history_client['place_destination_departure']
            
            # Exclui a coluna de hora e usa apenas as colunas necessárias
            history_table = df_history_client[['Data Compra', 'Trecho (Origem -> Destino)']]
            st.dataframe(history_table, use_container_width=True)

        else:
            st.warning(f"O ID do cliente '{customer_id}' não foi encontrado na base de dados para análise.")
    
    # --- Distribuição de Probabilidade de Compra ---
    st.markdown("---")
    st.subheader("Distribuição de Probabilidade de Compra")
    
    # Distribuição de probabilidade de Compra em 7 dias (Barra verde)
    st.markdown("### Distribuição de probabilidade de Compra em 7 dias")
    if not df_agg.empty:
        df_agg[['Prob Est 7d(%)', 'Prob Est 30d(%)']] = df_agg['avg_time_between_purchases'].apply(
            lambda x: pd.Series(calculate_exponential_probabilities(x))
        )
        
        bins = np.arange(0, 101, 5)
        hist_counts, bin_edges = np.histogram(df_agg['Prob Est 7d(%)'].dropna(), bins=bins)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(bin_edges[:-1], hist_counts, width=np.diff(bin_edges), color='green', edgecolor='black', align='edge')
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        plt.title("Distribuição de probabilidade de Compra em 7 dias")
        plt.xlabel("Probabilidade de Compra (%)")
        plt.ylabel("Número de Clientes")
        plt.xticks(bins, rotation=0, ha='center')
        st.pyplot(fig)
    else:
        st.info("Não foi possível gerar a distribuição de probabilidade de compra em 7 dias.")
    
    # Distribuição de probabilidade de Compra em 30 dias (Barra azul-clara)
    st.markdown("### Distribuição de probabilidade de Compra em 30 dias")
    if not df_agg.empty:
        bins = np.arange(0, 101, 5)
        hist_counts, bin_edges = np.histogram(df_agg['Prob Est 30d(%)'].dropna(), bins=bins)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(bin_edges[:-1], hist_counts, width=np.diff(bin_edges), color='skyblue', edgecolor='black', align='edge')
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        plt.title("Distribuição de probabilidade de Compra em 30 dias")
        plt.xlabel("Probabilidade de Compra (%)")
        plt.ylabel("Número de Clientes")
        plt.xticks(bins, rotation=0, ha='center')
        st.pyplot(fig)
    else:
        st.info("Não foi possível gerar a distribuição de probabilidade de compra em 30 dias.")
        
    st.markdown("---")
    st.subheader("Avaliação Visual do Modelo de Machine Learning")
    
    # Exibe a acurácia geral com o componente st.metric
    st.metric(label="Acurácia Geral do Modelo", value=f"{ml_accuracy*100:.2f}%")
    
    # Matriz de Confusão
    st.markdown("### Matriz de Confusão para as 10 Classes Mais Comuns")
    if len(np.unique(y_test_filtered)) > 1:
        plot_confusion_matrix_for_top_classes(y_test_filtered, y_pred, origin_le.classes_)
    else:
        st.info("Não foi possível gerar a matriz de confusão, pois há menos de duas classes no conjunto de teste filtrado.")
    
    # Gráfico de Métricas por Classe
    st.markdown("### Métricas de Classificação por Classe (Top 10)")
    if ml_report:
        plot_classification_metrics_bar_chart(ml_report)
    else:
        st.info("Não foi possível gerar o relatório de classificação.")
    
    st.markdown("---")
    st.subheader("Tabela de Clientes e Previsão da Próxima Compra")
    
    all_contacts = pd.DataFrame(df_full['fk_contact'].unique(), columns=['fk_contact'])
    df_final = pd.merge(all_contacts, df_agg, on='fk_contact', how='left')
    df_final = pd.merge(df_final, df_predictions, on='fk_contact', how='left')

    df_final[['Prob Est 7d(%)', 'Prob Est 30d(%)']] = df_final['avg_time_between_purchases'].apply(
        lambda x: pd.Series(calculate_exponential_probabilities(x))
    )
    
    df_final['Data Prevista'] = df_final.apply(
        lambda row: (row['date_purchase'] + timedelta(days=row['avg_time_between_purchases'])).strftime('%d/%m/%Y')
        if pd.notna(row['avg_time_between_purchases']) and pd.notna(row['date_purchase'])
        else 'N/A',
        axis=1
    )

    df_final['trecho_previsto_ml'] = df_final['trecho_previsto_ml'].fillna("Não é possível prever o trecho")
    df_final['avg_time_between_purchases'] = df_final['avg_time_between_purchases'].fillna(np.nan)

    tabela_exibicao = df_final[['fk_contact', 'Prob Est 7d(%)', 'Prob Est 30d(%)', 'avg_time_between_purchases', 'Data Prevista', 'trecho_previsto_ml']].copy()
    
    tabela_exibicao.rename(
        columns={
            'fk_contact': 'ID Cliente',
            'avg_time_between_purchases': 'Prev Est Próx Compra(dias)',
            'trecho_previsto_ml': 'Trecho Previsto (ML)'
        },
        inplace=True
    )

    tabela_exibicao['Prev Est Próx Compra(dias)'] = tabela_exibicao['Prev Est Próx Compra(dias)'].apply(lambda x: f"{int(x)}" if pd.notna(x) else 'N/A')
    tabela_exibicao['Prob Est 7d(%)'] = tabela_exibicao['Prob Est 7d(%)'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
    tabela_exibicao['Prob Est 30d(%)'] = tabela_exibicao['Prob Est 30d(%)'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
    
    if not tabela_exibicao.empty:
        st.dataframe(tabela_exibicao, use_container_width=True)
        st.download_button(
            label="📥 Baixar Tabela Completa",
            data=convert_df_to_csv(tabela_exibicao),
            file_name='previsoes_completas.csv',
            mime='text/csv'
        )
    else:
        st.warning("Não há dados de clientes para exibir.")