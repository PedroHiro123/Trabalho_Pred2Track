# Importação das bibliotecas necessárias
import streamlit as st
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import calendar
from PIL import Image


# Configuração da página do Streamlit
st.set_page_config(
    page_title="Dashboard de Análise de Clientes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Carregamento e processamento inicial dos dados (com cache)
@st.cache_data
def load_and_process_data(file_path):
    """
    Carrega o arquivo CSV e realiza o processamento inicial dos dados.
    Esta função é cacheada para evitar recarregar e reprocessar os dados a cada interação.
    """
    try:
        df = pd.read_csv(file_path)

        # Conversão de tipos de colunas para o formato correto
        df['date_purchase'] = pd.to_datetime(df['date_purchase'])
        df['nk_ota_localizer_id'] = df['nk_ota_localizer_id'].astype(str)
        df['fk_contact'] = df['fk_contact'].astype(str)

        # Criação de colunas auxiliares para análise temporal
        df['year_purchase'] = df['date_purchase'].dt.year
        df['month_purchase'] = df['date_purchase'].dt.month
        df['day_purchase'] = df['date_purchase'].dt.day
        df['day_of_week'] = df['date_purchase'].dt.day_name()
        df['hour_purchase'] = pd.to_datetime(df['time_purchase'], format='%H:%M:%S').dt.hour

        # Combinação de origem e destino para rota (ida e volta)
        df['route_departure'] = df['place_origin_departure'] + ' -> ' + df['place_destination_departure']
        df['route_return'] = df['place_origin_return'] + ' -> ' + df['place_destination_return']

        # Criação da lista de contatos para o filtro, garantindo que não há espaços extras
        all_contacts = ['Todos'] + sorted(df['fk_contact'].unique())

        return df, all_contacts
    except FileNotFoundError:
        st.error("Erro: O arquivo não foi encontrado. Por favor, verifique o caminho do arquivo.")
        return None, None

file_path = 'df_t_MODIFICADO.csv'
df, all_contacts = load_and_process_data(file_path)

# 2. Inicialização do estado de sessão para o filtro
if 'current_filter_id' not in st.session_state:
    st.session_state.current_filter_id = 'Todos'

if df is not None:
    
    # 3. Filtros interativos na barra lateral
    with st.sidebar:
        st.header("Filtros de Análise")
        
        # Filtro por Selectbox (opção principal)
        selected_contact_id = st.selectbox(
            "1. Selecione o ID do Cliente",
            options=all_contacts,
            index=0
        )
        
        st.markdown("---")
        
        # Filtro por Text Input e Botão (usando formulário)
        with st.form("filter_form"):
            st.markdown("**Ou digite o ID do Cliente:**")
            search_contact_id = st.text_input(
                "Digite o ID do Cliente",
                placeholder="Ex: CLI_123456789"
            )
            # st.form_submit_button se comporta como um botão normal, mas submete o formulário
            search_button = st.form_submit_button("Aplicar Filtro")
    
    # Lógica de atualização do filtro baseada na interação do usuário
    # Se o botão do formulário foi clicado (ou ENTER foi pressionado)
    if search_button:
        if search_contact_id.strip() in all_contacts:
            st.session_state.current_filter_id = search_contact_id.strip()
        elif search_contact_id.strip() == '':
            st.session_state.current_filter_id = 'Todos'
        else:
            st.error("ID de cliente não encontrado. Verifique o ID digitado.")
    # Se o valor do selectbox for alterado, ele sobrescreve o filtro
    elif selected_contact_id != st.session_state.current_filter_id:
        st.session_state.current_filter_id = selected_contact_id

    # Lógica final para filtrar o DataFrame usando o estado de sessão
    if st.session_state.current_filter_id != 'Todos':
        filtered_df = df[df['fk_contact'] == st.session_state.current_filter_id].copy()
        current_filter_label = st.session_state.current_filter_id
    else:
        filtered_df = df.copy()
        current_filter_label = "Todos os Clientes"
        
    st.sidebar.info(f"Filtro atual: **{current_filter_label}**")

    # Título do Dashboard e Logotipo alinhado à direita
    col_title, col_logo = st.columns([0.8, 0.2])

    with col_title:
        st.title(f"Monitoramento do Comportamento de Compra: {current_filter_label}")
    with col_logo:
        try:
            image_path = 'ClickBus_logo.png'
            image = Image.open(image_path)
            st.image(image, width=150)
        except FileNotFoundError:
            st.error("Erro: Arquivo do logo não encontrado. Verifique o caminho.")
    
    st.markdown("Análise do comportamento de Compra para Segmentação de Clientes e Direcionamento de Estratégias de Marketing.")
    st.markdown("---")

    # 4. Análise de Comportamento de Compra (RFM)
    # Esta seção só será exibida se o filtro for para "Todos os Clientes"
    if current_filter_label == "Todos os Clientes":
        st.header("Análise de Comportamento de Compra (RFM)")
        
        if not filtered_df.empty:
            
            # Cálculo da Tabela RFM
            ref_date = filtered_df['date_purchase'].max() + dt.timedelta(days=1)
            rfm_table = filtered_df.groupby('fk_contact').agg(
                Recency=('date_purchase', lambda x: (ref_date - x.max()).days),
                Frequency=('nk_ota_localizer_id', 'count'),
                Monetary=('gmv_success', 'sum')
            ).reset_index()

            rfm_table['Monetary'] = rfm_table['Monetary'].round(2)

            # Ajustando o número de quantis
            num_unique_r = len(rfm_table['Recency'].unique())
            num_unique_f = len(rfm_table['Frequency'].unique())
            num_unique_m = len(rfm_table['Monetary'].unique())

            q_r = min(5, num_unique_r)
            q_f = min(5, num_unique_f)
            q_m = min(5, num_unique_m)

            if q_r > 1 and q_f > 1 and q_m > 1:
                # Definindo os scores RFM de forma dinâmica
                rfm_table.loc[:, 'R_Score'] = pd.qcut(rfm_table['Recency'], q=q_r, labels=False, duplicates='drop')
                rfm_table.loc[:, 'F_Score'] = pd.qcut(rfm_table['Frequency'], q=q_f, labels=False, duplicates='drop')
                rfm_table.loc[:, 'M_Score'] = pd.qcut(rfm_table['Monetary'], q=q_m, labels=False, duplicates='drop')
                
                # Padronizando os scores para a escala de 1 a 5
                rfm_table.loc[:, 'R_Score'] = 5 - rfm_table['R_Score']
                rfm_table.loc[:, 'F_Score'] = rfm_table['F_Score'] + 1
                rfm_table.loc[:, 'M_Score'] = rfm_table['M_Score'] + 1
                
                # Função para segmentação de clientes com base nos scores RFM
                def rfm_segment(row):
                    r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
                    if r in [4, 5] and f in [4, 5] and m in [4, 5]:
                        return 'Champions'
                    elif r in [4, 5] and f in [3, 4] and m in [3, 4]:
                        return 'Loyal Customers'
                    elif r in [4, 5] and f in [1, 2]:
                        return 'Potential Loyalists'
                    elif r in [3] and f in [3]:
                        return 'Need Attention'
                    elif r in [1, 2] and f in [4, 5]:
                        return 'At Risk'
                    elif r in [1, 2] and f in [1, 2]:
                        return 'Hibernating'
                    else:
                        return 'Others'

                rfm_table.loc[:, 'RFM_Segment'] = rfm_table.apply(rfm_segment, axis=1)

                st.markdown("### Segmentação de Clientes (RFM)")
                
                # Gráfico de Distribuição de Clientes por Segmento
                segment_order = rfm_table['RFM_Segment'].value_counts().index
                fig_segment, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(
                    data=rfm_table,
                    x='RFM_Segment',
                    hue='RFM_Segment',
                    ax=ax,
                    palette='viridis',
                    order=segment_order,
                    legend=False
                )
                ax.set_title('Distribuição de Clientes por Segmento RFM')
                ax.set_xlabel('Segmento RFM')
                ax.set_ylabel('Número de Clientes')
                plt.xticks(rotation=45, ha='right')
                # Adiciona separador numérico no eixo Y
                ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
                st.pyplot(fig_segment)

                st.markdown("### Tabela RFM com Segmentos")
                st.dataframe(rfm_table, use_container_width=True)

                # Adiciona o botão de download da tabela RFM
                csv_data = rfm_table.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Baixar Tabela de Segmentos RFM (CSV)",
                    data=csv_data,
                    file_name='rfm_segments.csv',
                    mime='text/csv',
                )

                # Seção de informações semânticas dos clusters
                st.markdown("---")
                st.header("Informações Semânticas dos Clusters RFM")
                st.markdown("""
                Esta seção oferece uma visão detalhada sobre o significado de cada segmento de cliente,
                baseado na análise RFM (Recência, Frequência e Monetário).
                As pontuações são de 1 a 5, onde 5 é a melhor.
                """)
        
                clusters_info = {
                    "Champions": {
                        "desc": "São os melhores clientes. Compraram recentemente, com frequência e gastaram mais. São os mais leais e rentáveis.",
                        "r_f_m": "(R: 4-5, F: 4-5, M: 4-5)",
                        "estrategia": "Recompense-os. Podem ser embaixadores da marca. Lance novos produtos e serviços para eles."
                    },
                    "Loyal Customers": {
                        "desc": "Compram com frequência e gastaram uma boa quantia. Respondem bem a promoções.",
                        "r_f_m": "(R: 4-5, F: 3-4, M: 3-4)",
                        "estrategia": "Ofereça produtos de valor mais alto. Peça feedbacks. Interaja com eles."
                    },
                    "Potential Loyalists": {
                        "desc": "Compraram recentemente, mas ainda não com muita frequência. Têm potencial para se tornarem fiéis.",
                        "r_f_m": "(R: 4-5, F: 1-2, M: 1-5)",
                        "estrategia": "Ofereça um programa de fidelidade. Recomende produtos. Ajude a aumentar sua frequência de compra."
                    },
                    "Need Attention": {
                        "desc": "As compras deles foram há um tempo, mas compraram com uma boa frequência e gastaram bem. Precisam de atenção para não se tornarem clientes em risco.",
                        "r_f_m": "(R: 3, F: 3, M: 3)",
                        "estrategia": "Envie campanhas de reengajamento personalizadas. Ofereça promoções limitadas ou benefícios especiais."
                    },
                    "At Risk": {
                        "desc": "Compraram muito (frequência) e gastaram muito (monetário) no passado, mas não compram há muito tempo. Podem estar migrando para a concorrência.",
                        "r_f_m": "(R: 1-2, F: 4-5, M: 4-5)",
                        "estrategia": "Tente reconquistá-los com e-mails personalizados, ofertas de aniversário ou promoções especiais."
                    },
                    "Hibernating": {
                        "desc": "As últimas compras foram há muito tempo e eles compraram pouco. É o segmento mais difícil de recuperar.",
                        "r_f_m": "(R: 1-2, F: 1-2, M: 1-2)",
                        "estrategia": "Considere o mínimo esforço para recuperá-los. Use campanhas de baixo custo ou automáticas para tentar um reengajamento."
                    },
                    "Others": {
                        "desc": "Clientes que não se encaixam claramente nos segmentos principais. Podem ter comportamentos de compra variados.",
                        "r_f_m": "(R: Outros, F: Outros, M: Outros)",
                        "estrategia": "Monitore-os individualmente ou use estratégias genéricas de marketing para tentar movê-los para um dos segmentos principais."
                    }
                }
        
                for segment_name, info in clusters_info.items():
                    with st.expander(f"**{segment_name}**"):
                        st.markdown(f"**Descrição:** {info['desc']}")
                        st.markdown(f"**Critério RFM:** {info['r_f_m']}")
                        st.markdown(f"**Estratégias Sugeridas:** {info['estrategia']}")

                # Gráfico de Distribuição de Clientes por Segmento
                st.markdown("### Distribuição de Clientes por Segmento")
                fig_scatter, ax_scatter = plt.subplots(figsize=(12, 8))
                sns.scatterplot(
                    data=rfm_table,
                    x='Recency',
                    y='Frequency',
                    size='Monetary',
                    hue='RFM_Segment',
                    sizes=(20, 200),
                    palette='viridis',
                    hue_order=segment_order,
                    alpha=0.6,
                    edgecolor='w',
                    ax=ax_scatter
                )
                ax_scatter.set_title('Análise RFM: Recência vs. Frequência (Tamanho da bolha = Monetário)', fontsize=16)
                ax_scatter.set_xlabel('Recência (dias desde a última compra)', fontsize=12)
                ax_scatter.set_ylabel('Frequência (total de compras)', fontsize=12)
                ax_scatter.legend(title='Segmento RFM', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax_scatter.grid(True)
                # Adiciona separador numérico nos eixos X e Y
                ax_scatter.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
                ax_scatter.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
                plt.tight_layout()
                st.pyplot(fig_scatter)


            else:
                st.warning("Dados insuficientes para calcular os scores RFM.")
            
            st.markdown("---")

    # 5. Métricas Chave de Negócio (KPIs)
    st.header("Métricas Chave de Negócio (KPIs)")
    col1, col2, col3 = st.columns(3)
    
    # Cálculo dos KPIs principais
    num_clientes = filtered_df['fk_contact'].nunique()
    
    # Receita de todos os clientes para o cálculo do Ticket Médio (ARPU)
    receita_todos_clientes = df['gmv_success'].sum()
    total_tickets_vendidos = df['nk_ota_localizer_id'].nunique()
    
    # Cálculo do Ticket Médio (ARPU) para exibir sempre
    ticket_medio_arpu = receita_todos_clientes / total_tickets_vendidos if total_tickets_vendidos > 0 else 0
    
    col1.metric("Clientes Únicos", f"{num_clientes:,.0f}")
    
    # Exibe o Ticket Médio (ARPU) independentemente do filtro
    col2.metric("Ticket Médio (ARPU)", f"R$ {ticket_medio_arpu:,.2f}")
    
    # Exibe o Ticket Médio do Cliente somente se um cliente for selecionado
    if current_filter_label != "Todos os Clientes":
        receita_cliente_selecionado = filtered_df['gmv_success'].sum()
        tickets_cliente_selecionado = filtered_df['nk_ota_localizer_id'].nunique()
        ticket_medio_cliente = receita_cliente_selecionado / tickets_cliente_selecionado if tickets_cliente_selecionado > 0 else 0
        col3.metric("Ticket Médio do Cliente", f"R$ {ticket_medio_cliente:,.2f}")
    else:
        col3.metric("Ticket Médio do Cliente", "N/A")

    # Taxa de Retenção e Churn Anual - exibida somente se o filtro for "Todos"
    if current_filter_label == "Todos os Clientes":
        st.markdown("### Taxa de Retenção e Churn Anual")
        
        customers_by_year = filtered_df.groupby(filtered_df['date_purchase'].dt.to_period('Y'))['fk_contact'].nunique()
        
        if len(customers_by_year) > 1:
            retention_rate = []
            churn_rate = []
            
            for i in range(1, len(customers_by_year)):
                current_year_customers = filtered_df[filtered_df['year_purchase'] == customers_by_year.index[i].year]['fk_contact'].unique()
                previous_year_customers = filtered_df[filtered_df['year_purchase'] == customers_by_year.index[i-1].year]['fk_contact'].unique()
                
                retained_customers = len(set(current_year_customers) & set(previous_year_customers))
                
                retention = retained_customers / len(previous_year_customers) * 100
                churn = 100 - retention
                
                retention_rate.append(retention)
                churn_rate.append(churn)

            avg_retention = np.mean(retention_rate) if retention_rate else 0
            avg_churn = np.mean(churn_rate) if churn_rate else 0
            
            st.metric("Taxa de Retenção Média Anual", f"{avg_retention:,.2f}%")
            st.metric("Taxa de Churn Média Anual", f"{avg_churn:,.2f}%")
        else:
            st.info("Dados insuficientes para calcular a taxa de retenção e churn anual.")
    
    st.markdown("---")
    
    # 6. Análise de Frequência e Preferências de Compra
    st.header("Análise de Frequência e Preferências")

    col4, col5 = st.columns(2)

    if current_filter_label == "Todos os Clientes":
        if 'rfm_table' in locals() and not rfm_table.empty:
            avg_frequency = rfm_table['Frequency'].mean()
            col4.metric("Frequência Média de Compra por Cliente", f"{avg_frequency:,.1f} compras")
        else:
            col4.metric("Frequência Média de Compra por Cliente", "N/A")
    else:
        if not filtered_df.empty:
            avg_frequency = len(filtered_df)
            col4.metric("Total de Compras", f"{avg_frequency:,.0f} compras")

    avg_tickets = filtered_df['total_tickets_quantity_success'].mean()
    col5.metric("Média de Passagens por Compra", f"{avg_tickets:,.1f} passagens")

    st.markdown("### Compras por Período de Tempo")
    
    fig_year, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='year_purchase', ax=ax)
    ax.set_title('Compras por Ano')
    ax.set_xlabel('Ano')
    ax.set_ylabel('Número de Compras')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Adiciona separador numérico no eixo Y
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig_year)
    
    filtered_df.loc[:, 'month_name'] = filtered_df['month_purchase'].apply(lambda x: calendar.month_name[x])
    monthly_sales = filtered_df.groupby('month_name').agg(
        total_compras=('nk_ota_localizer_id', 'count')
    ).reset_index()

    month_order = [calendar.month_name[i] for i in range(1, 13)]
    monthly_sales['month_name'] = pd.Categorical(monthly_sales['month_name'], categories=month_order, ordered=True)
    monthly_sales = monthly_sales.sort_values('month_name')

    fig_month, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=monthly_sales, x='month_name', y='total_compras', ax=ax)
    ax.set_title('Compras por Mês (Acumulado Anual)')
    ax.set_xlabel('Mês')
    ax.set_ylabel('Número de Compras')
    plt.xticks(rotation=45, ha='right')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Adiciona separador numérico no eixo Y
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig_month)
    
    fig_day_of_month, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=filtered_df, x='day_purchase', ax=ax)
    ax.set_title('Compras por Dia do Mês')
    ax.set_xlabel('Dia do Mês')
    ax.set_ylabel('Número de Compras')
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Adiciona separador numérico no eixo Y
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig_day_of_month)
    
    days_order = ['Domingo', 'Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 'Sexta-feira', 'Sábado']
    filtered_df.loc[:, 'day_of_week_pt'] = filtered_df['day_of_week'].map({
        'Sunday': 'Domingo', 'Monday': 'Segunda-feira', 'Tuesday': 'Terça-feira', 
        'Wednesday': 'Quarta-feira', 'Thursday': 'Quinta-feira', 
        'Friday': 'Sexta-feira', 'Saturday': 'Sábado'
    })
    
    fig_day, ax_day = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='day_of_week_pt', ax=ax_day, order=days_order)
    ax_day.set_title('Compras por Dia da Semana')
    ax_day.set_xlabel('Dia da Semana')
    ax_day.set_ylabel('Número de Compras')
    plt.xticks(rotation=45, ha='right')
    ax_day.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Adiciona separador numérico no eixo Y
    ax_day.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig_day)
    
    fig_hour, ax_hour = plt.subplots(figsize=(10, 6))
    sns.countplot(data=filtered_df, x='hour_purchase', ax=ax_hour)
    ax_hour.set_title('Compras por Hora do Dia')
    ax_hour.set_xlabel('Hora do Dia')
    ax_hour.set_ylabel('Número de Compras')
    
    ax_hour.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Adiciona separador numérico no eixo Y
    ax_hour.yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig_hour)
    
    st.markdown("---")
    
    # 7. Novo gráfico: Top 100 Clientes por Receita
    st.header("Top 100 Clientes por Receita")
    
    if not filtered_df.empty:
        top_clientes_receita = filtered_df.groupby('fk_contact')['gmv_success'].sum().nlargest(100).reset_index()
        top_clientes_receita.columns = ['ID do Cliente', 'Receita (R$)']
        
        num_clientes_rev = len(top_clientes_receita)
        fig_height_rev = max(5, num_clientes_rev * 0.2)
        
        fig_clientes_rev, ax_clientes_rev = plt.subplots(figsize=(10, fig_height_rev))
        sns.barplot(data=top_clientes_receita, x='Receita (R$)', y='ID do Cliente', ax=ax_clientes_rev)
        ax_clientes_rev.set_title(f'Top {num_clientes_rev} Clientes por Receita')
        ax_clientes_rev.set_xlabel('Receita (R$)')
        ax_clientes_rev.set_ylabel('ID do Cliente')
        # Adiciona separador numérico no eixo X
        ax_clientes_rev.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
        st.pyplot(fig_clientes_rev)
    else:
        st.warning("Não há dados de receita para o filtro selecionado.")
        
    st.markdown("---")
    
    # 8. Análise de Trechos e Viações
    st.header("Análise de Trechos e Viações")
    
    top_routes_revenue = filtered_df.groupby('route_departure')['gmv_success'].sum().nlargest(100).reset_index()
    top_routes_revenue.columns = ['Rota', 'Receita']
    
    num_routes_rev = len(top_routes_revenue)
    fig_height_rev = max(5, num_routes_rev * 0.2)
    
    fig_routes_rev, ax_routes_rev = plt.subplots(figsize=(10, fig_height_rev))
    sns.barplot(data=top_routes_revenue, x='Receita', y='Rota', ax=ax_routes_rev)
    ax_routes_rev.set_title(f'Top {num_routes_rev} Rotas por Receita')
    ax_routes_rev.set_xlabel('Receita (R$)')
    ax_routes_rev.set_ylabel('Rota')
    # Adiciona separador numérico no eixo X
    ax_routes_rev.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig_routes_rev)

    top_routes_count = filtered_df['route_departure'].value_counts().nlargest(100).reset_index()
    top_routes_count.columns = ['Rota', 'Número de Compras']

    num_routes_count = len(top_routes_count)
    fig_height_count = max(5, num_routes_count * 0.2)
    
    fig_routes_count, ax_routes_count = plt.subplots(figsize=(10, fig_height_count))
    sns.barplot(data=top_routes_count, x='Número de Compras', y='Rota', ax=ax_routes_count)
    ax_routes_count.set_title(f'Top {num_routes_count} Rotas por Número de Compras')
    ax_routes_count.set_xlabel('Número de Compras')
    ax_routes_count.set_ylabel('Rota')
    # Adiciona separador numérico no eixo X
    ax_routes_count.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig_routes_count)

    top_origins_departure = filtered_df['place_origin_departure'].value_counts().nlargest(10).reset_index()
    top_origins_departure.columns = ['Origem', 'Número de Compras']
    
    fig_origin_dep, ax_origin_dep = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_origins_departure, x='Número de Compras', y='Origem', ax=ax_origin_dep)
    ax_origin_dep.set_title('Origens Mais Frequentes (ida)')
    ax_origin_dep.set_xlabel('Número de Compras')
    ax_origin_dep.set_ylabel('Origem')
    # Adiciona separador numérico no eixo X
    ax_origin_dep.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))
    st.pyplot(fig_origin_dep)
    
    top_origins_return = filtered_df['place_origin_return'].dropna().value_counts().nlargest(10).reset_index()
    top_origins_return.columns = ['Origem', 'Número de Compras']
    
    fig_origin_ret, ax_origin_ret = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top_origins_return, x='Número de Compras', y='Origem', ax=ax_origin_ret)
    ax_origin_ret.set_title('Origens Mais Frequentes (retorno)')
    ax_origin_ret.set_xlabel('Número de Compras')
    ax_origin_ret.set_ylabel('Origem')
    st.pyplot(fig_origin_ret)
    
    
    # Prepara os dados de receita para o gráfico de pizza
    df_viacao_dep_filtered = filtered_df[['fk_departure_ota_bus_company', 'gmv_success']].rename(columns={'fk_departure_ota_bus_company': 'viacao'}).copy()
    df_viacao_ret_filtered = filtered_df[['fk_return_ota_bus_company', 'gmv_success']].rename(columns={'fk_return_ota_bus_company': 'viacao'}).copy()
    
    # Combina os DataFrames de ida e volta e remove valores nulos
    df_viacao_total_filtered = pd.concat([df_viacao_dep_filtered, df_viacao_ret_filtered]).dropna(subset=['viacao'])
    
    if not df_viacao_total_filtered.empty:
        # Soma a receita por viação
        receita_por_viacao = df_viacao_total_filtered.groupby('viacao')['gmv_success'].sum().nlargest(10)
        
        # Cria o rótulo para o gráfico de pizza, incluindo o nome e a porcentagem
        labels = receita_por_viacao.index
        sizes = receita_por_viacao.values
        total_revenue = sizes.sum()
        
        # Função para formatar o rótulo com porcentagem
        def format_pct(pct):
            return f'{pct:.1f}%' if pct > 0 else ''
        
        # Cria a figura e o eixo
        fig_viacao, ax_viacao = plt.subplots(figsize=(10, 8)) # Aumentei um pouco a largura da figura
        
        # O gráfico de pizza é criado
        wedges, texts, autotexts = ax_viacao.pie(
            sizes,
            autopct=format_pct,
            startangle=90,
            pctdistance=1.05 # Valor alterado para aproximar os percentuais
        )
        
        # Título do gráfico com tamanho de fonte padronizado
        ax_viacao.set_title('Participação Percentual na Receita por Viação', fontsize=14)
        
        # Cria a legenda fora do gráfico, ajustando a posição
        ax_viacao.legend(
            labels,
            title="Viação",
            loc="center left",
            bbox_to_anchor=(1.05, 0, 0.5, 1) # Aumentei o primeiro valor (x) para afastar a caixa
        )
        
        # Garante que o círculo seja desenhado corretamente
        ax_viacao.axis('equal')
        
        # Ajusta o layout para evitar sobreposição
        plt.tight_layout()
        
        st.pyplot(fig_viacao)
    else:
        st.warning("Não há dados de viação para o cliente ou filtro selecionado.")
    # -----------------------------------------------------------------------------------

else:
    st.warning("O filtro selecionado não possui dados. Por favor, ajuste sua seleção.")
