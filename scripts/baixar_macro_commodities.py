import os
import yfinance as yf

# Lista de tickers mapeados
tickers = {
    'petroleo_brent': 'BZ=F',
    'dxy': 'DX-Y.NYB',
    'ewz': 'EWZ',
    'ouro': 'GC=F',
    'sp500': '^GSPC',
    'nasdaq': '^IXIC',
    'us10y': '^TNX',
    'vix': '^VIX'
}

print("Baixando dados históricos do Yahoo Finance...")
# Baixar todos de uma vez (usando 'Close' em vez de 'Adj Close')
dados = yf.download(list(tickers.values()), start="2015-01-01", end="2025-12-31")['Close']

# Renomear as colunas para o nome amigável padronizado
mapa_reverso = {v: k for k, v in tickers.items()}
dados.rename(columns=mapa_reverso, inplace=True)

# Garantir que a coluna de data é nomeada como 'data'
dados.index.name = 'data'

# Determinar caminho correto relativo à pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.abspath(os.path.join(script_dir, '../data/dados_macro_commodities.csv'))

os.makedirs(os.path.dirname(out_path), exist_ok=True)

# Salvar em CSV padronizado (com delimitador vírgula e formato UTF-8)
dados.to_csv(out_path, encoding='utf-8')
print(f"Concluído! Arquivo '{out_path}' gerado com sucesso.")
