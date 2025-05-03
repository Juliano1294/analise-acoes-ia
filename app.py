import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout='wide', page_title="Análise de Ações com IA")
st.title("📈 Análise Técnica com IA para Ações")

# Sidebar - entrada do usuário
with st.sidebar:
    st.header("Configurações")
    ticker = st.text_input("Ticker da ação", value="PETR4.SA")
    periodo = st.selectbox("Período", ['3mo', '6mo', '1y', '2y'], index=2)

# Funções
@st.cache_data
def preparar_dados(ticker, periodo):
    df = yf.download(ticker, period=periodo)

    if df.empty:
        st.error("Erro: Nenhum dado retornado. Verifique o ticker ou o período selecionado.")
        return pd.DataFrame()

    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = RSIIndicator(df['Close']).rsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['Target'] = 0
    df['Future Close'] = df['Close'].shift(-1)
    df.loc[df['Future Close'] > df['Close'], 'Target'] = 1
    df.loc[df['Future Close'] < df['Close'], 'Target'] = -1
    df.dropna(inplace=True)
    return df

def treinar_modelo(df):
    X = df[['MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal']]
    y = df['Target']
    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(X, y)
    df['Previsao'] = modelo.predict(X)
    return df, modelo

def simular_operacoes(df):
    saldo = 10000
    posicao = 0
    historico = []

    for i in range(len(df)):
        preco = df['Close'].iloc[i]
        sinal = df['Previsao'].iloc[i]
        data = df.index[i]

        if sinal == 1 and saldo >= preco:
            qtd = saldo // preco
            saldo -= qtd * preco
            posicao += qtd
            historico.append((data, 'COMPRA', preco, qtd))
        elif sinal == -1 and posicao > 0:
            saldo += posicao * preco
            historico.append((data, 'VENDA', preco, posicao))
            posicao = 0

    valor_final = saldo + posicao * df['Close'].iloc[-1]
    return historico, valor_final

def plotar(df):
    st.markdown("### Gráfico com Sinais de Compra e Venda")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['Close'], label='Preço', color='blue')
    ax.plot(df['MA20'], label='MA20', linestyle='--', alpha=0.7)
    ax.plot(df['MA50'], label='MA50', linestyle='--', alpha=0.7)

    compras = df[df['Previsao'] == 1]
    vendas = df[df['Previsao'] == -1]

    ax.plot(compras.index, compras['Close'], '^', markersize=10, color='green', label='Compra')
    ax.plot(vendas.index, vendas['Close'], 'v', markersize=10, color='red', label='Venda')

    ax.set_title("Sinais de Compra/Venda com Indicadores Técnicos")
    ax.legend()
    st.pyplot(fig)

def exibir_rsi(df):
    st.markdown("### RSI")
    fig, ax = plt.subplots()
    ax.plot(df['RSI'], label='RSI', color='orange')
    ax.axhline(70, color='red', linestyle='--')
    ax.axhline(30, color='green', linestyle='--')
    ax.set_title("Índice de Força Relativa (RSI)")
    st.pyplot(fig)

# Execução
df = preparar_dados(ticker, periodo)
if df.empty:
    st.stop()

df, modelo = treinar_modelo(df)
historico, valor_final = simular_operacoes(df)

# Layout em colunas
col1, col2 = st.columns(2)

with col1:
    plotar(df)

with col2:
    exibir_rsi(df)
    st.markdown("### Resultado da Carteira")
    st.success(f"Valor final: R$ {valor_final:,.2f}")

# Tabela de operações
st.markdown("### Histórico de Operações Realizadas")
st.dataframe(pd.DataFrame(historico, columns=['Data', 'Operação', 'Preço', 'Quantidade']), use_container_width=True)