import os
import pandas as pd
import pyfredapi as pf
from dotenv import load_dotenv

# 1. SETUP FRED
# Determinar caminho correto relativo à pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.abspath(os.path.join(script_dir, '../.env'))

# Carrega as variáveis de ambiente do arquivo .env da raiz do projeto
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv('FRED_API_KEY')

if not api_key:
    raise ValueError(f"Erro: Chave 'FRED_API_KEY' não encontrada no arquivo .env no caminho: {dotenv_path}")

# Configura a chave do FRED globalmente para o pyfredapi
os.environ['FRED_API_KEY'] = api_key

print("Conectando à API do FRED...")

# 2. DEFINIÇÃO DAS SÉRIES DOS EUA
series_fred = {
    'CPI_EUA': 'CPIAUCSL',        # Inflação Americana
    'PIB_EUA': 'GDP',             # Produto Interno Bruto
    'Payroll_NFP': 'PAYEMS',      # Criação de Empregos Não-Agrícolas
    'Juros_Fed': 'DFEDTARU'       # Taxa de Juros (Limite Superior) - FOMC
}

df_eua_list = []

# 3. EXTRAÇÃO DOS DADOS
for evento, codigo in series_fred.items():
    print(f"Baixando histórico de: {evento}...")
    try:
        # Puxa a série temporal inteira
        serie = pf.get_series(series_id=codigo)
        
        # Filtra estritamente para o período solicitado: 2015 a 2025
        mask = (serie['date'] >= '2015-01-01') & (serie['date'] <= '2025-12-31')
        df_temp = serie.loc[mask].copy()
        
        # FRED retorna 'date' e 'value'.
        df_temp = df_temp[['date']].rename(columns={'date': 'data'})
        df_temp['evento'] = evento
        
        df_eua_list.append(df_temp)
    except Exception as e:
        print(f"  -> Erro ao baixar {evento}: {e}")

# 4. CONSOLIDAÇÃO E EXPORTAÇÃO
if df_eua_list:
    # Junta todas as tabelas baixadas em uma só
    dados_eua = pd.concat(df_eua_list, ignore_index=True)
    
    # Ordena para ficar cronológico e organizado por evento
    dados_eua.sort_values(by=['evento', 'data'], inplace=True)
    
    # Mantém as colunas padronizadas
    df_final = dados_eua[['data', 'evento']]
    
    # Salva no arquivo final padronizado
    out_path = os.path.abspath(os.path.join(script_dir, '../data/datas_eventos_g6_eua.csv'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_final.to_csv(out_path, index=False, encoding='utf-8')
    
    print(f"\nSucesso! Arquivo '{out_path}' gerado com as datas do mercado americano.")
else:
    print("\nFalha: Nenhum dado foi baixado. Verifique sua conexão e chave de API.")
