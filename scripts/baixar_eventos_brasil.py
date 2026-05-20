import os
import pandas as pd
from bcb import Expectativas, sgs

print("Baixando dados do Banco Central do Brasil...")

# 1. Relatório Focus (Datas de publicação)
try:
    em = Expectativas()
    ep = em.get_endpoint('ExpectativasMercadoAnuais')
    focus = ep.query().filter(ep.Data >= '2015-01-01').select(ep.Data).collect()
    datas_focus = pd.to_datetime(focus['Data']).drop_duplicates().sort_values().dt.strftime('%Y-%m-%d').reset_index(drop=True)
except Exception as e:
    print(f"Erro ao baixar dados do Focus: {e}")
    datas_focus = []

# 2. COPOM (Decisão de Juros BR)
try:
    # Dividindo em duas consultas para contornar a restrição de 10 anos da API do BCB
    selic_pt1 = sgs.get({'selic_meta': 432}, start='2015-01-01', end='2019-12-31')
    selic_pt2 = sgs.get({'selic_meta': 432}, start='2020-01-01', end='2025-12-31')
    selic = pd.concat([selic_pt1, selic_pt2])
    datas_copom = selic.index.to_series().dt.strftime('%Y-%m-%d').reset_index(drop=True)
except Exception as e:
    print(f"Erro ao baixar dados do COPOM: {e}")
    datas_copom = []

# 3. Vencimento WIN (Última quinta-feira do mês)
datas_win = pd.date_range(start='2015-01-01', end='2025-12-31', freq='WOM-4THU').strftime('%Y-%m-%d')

# Salvando os resultados
df_eventos_br = pd.DataFrame({
    'data_focus': pd.Series(datas_focus),
    'data_copom': pd.Series(datas_copom),
    'data_vencimento_win': pd.Series(datas_win)
})

# Determinar caminho correto relativo à pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.abspath(os.path.join(script_dir, '../data/datas_eventos_brasil.csv'))

os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_eventos_br.to_csv(out_path, index=False, encoding='utf-8')
print(f"Concluído! Arquivo '{out_path}' gerado com sucesso.")
