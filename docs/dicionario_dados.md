# Dicionário de Dados do Projeto

Este documento serve como um guia explicativo para facilitar o uso dos arquivos de dados do projeto. Todos os arquivos CSV foram padronizados para garantir consistência e facilidade de importação em pipelines de ciência de dados (por exemplo, usando Python/Pandas).

---

## 📐 Padrão Geral dos Arquivos CSV

Todos os arquivos armazenados na pasta `data/` seguem estritamente as seguintes regras:
- **Codificação (Encoding)**: `utf-8`
- **Delimitador**: `,` (vírgula)
- **Separador Decimal**: `.` (ponto)
- **Nome da Coluna de Data**: Sempre **`data`** (ou `data_...` no caso de múltiplas séries emparelhadas).
- **Formato das Datas**: Padrão ISO `YYYY-MM-DD` (ex: `2026-05-19`).
- **Nomes de Colunas**: Formato `snake_case` (letras minúsculas, sem espaços, acentos ou caracteres especiais).

---

## 📂 Detalhamento dos Arquivos

### 1. `minerio_ferro_futuros.csv`
Histórico de preços e volumes diários de contratos futuros do minério de ferro refinado 62% Fe CFR.
- **Origem**: Ingestão de dados históricos (ex: Investing.com).
- **Esquema de Colunas**:

| Nome da Coluna | Tipo de Dado | Descrição | Exemplo |
| :--- | :--- | :--- | :--- |
| `data` | `str` (Date) | Data do registro no formato `YYYY-MM-DD` | `2026-05-19` |
| `ultimo` | `float` | Preço de fechamento da sessão de negociação em USD/tonelada | `110.33` |
| `abertura` | `float` | Preço de abertura da sessão de negociação em USD/tonelada | `110.33` |
| `maxima` | `float` | Preço máximo atingido durante a sessão em USD/tonelada | `110.33` |
| `minima` | `float` | Preço mínimo atingido durante a sessão em USD/tonelada | `110.33` |
| `volume` | `float` | Volume de contratos negociados (sufixos 'K' foram convertidos, ex: 30.0) | `30.0` |
| `variacao_pct` | `float` | Variação percentual diária em relação ao dia anterior (ex: -0.19 significa -0.19%) | `-0.19` |

---

### 2. `ipca.csv`
Histórico da inflação oficial brasileira medida pela variação percentual mensal do IPCA.
- **Origem**: IBGE (via SGS/Banco Central do Brasil).
- **Esquema de Colunas**:

| Nome da Coluna | Tipo de Dado | Descrição | Exemplo |
| :--- | :--- | :--- | :--- |
| `data` | `str` (Date) | Data do primeiro dia do mês de referência no formato `YYYY-MM-01` | `2015-01-01` |
| `ipca_var_mensal_pct` | `float` | Variação percentual do IPCA no mês (ex: 1.24 significa 1.24%) | `1.24` |

---

### 3. `selic.csv`
Histórico da Taxa Selic diária (Taxa Selic Meta definida pelo COPOM).
- **Origem**: SGS/Banco Central do Brasil (Série 432).
- **Esquema de Colunas**:

| Nome da Coluna | Tipo de Dado | Descrição | Exemplo |
| :--- | :--- | :--- | :--- |
| `data` | `str` (Date) | Data do registro no formato `YYYY-MM-DD` | `2015-01-01` |
| `selic_meta_ano_pct` | `float` | Taxa de juros anualizada definida pelo Copom (ex: 11.75 significa 11.75% a.a.) | `11.75` |

---

### 4. `pib.csv`
Histórico anual do Produto Interno Bruto (PIB) do Brasil medido em milhões de dólares correntes.
- **Origem**: BCB-Depec (Série SGS 7324).
- **Esquema de Colunas**:

| Nome da Coluna | Tipo de Dado | Descrição | Exemplo |
| :--- | :--- | :--- | :--- |
| `data` | `str` (Date) | Data do primeiro dia do ano de referência no formato `YYYY-01-01` | `2015-01-01` |
| `pib_usd_milhoes` | `float` | Valor total do PIB anual em milhões de dólares correntes | `1796167.58` |

---

### 5. `dados_macro_commodities.csv`
Séries históricas diárias de ativos globais, indicadores macroeconômicos e commodities de relevância global.
- **Origem**: Yahoo Finance (baixado automaticamente via script).
- **Esquema de Colunas**:

| Nome da Coluna | Tipo de Dado | Descrição | Ticker de Origem |
| :--- | :--- | :--- | :--- |
| `data` | `str` (Date) | Data do pregão no formato `YYYY-MM-DD` | `-` |
| `petroleo_brent` | `float` | Preço de fechamento do Petróleo Brent futuro (USD/barril) | `BZ=F` |
| `dxy` | `float` | Índice DXY (Dollar Index) - força do dólar ante cesta de moedas | `DX-Y.NYB` |
| `ewz` | `float` | Preço de fechamento do ETF iShares MSCI Brazil (EWZ) em Nova York | `EWZ` |
| `ouro` | `float` | Preço de fechamento do Contrato Futuro de Ouro (USD/onça-troy) | `GC=F` |
| `sp500` | `float` | Preço de fechamento do Índice S&P 500 | `^GSPC` |
| `nasdaq` | `float` | Preço de fechamento do Índice Nasdaq Composite | `^IXIC` |
| `us10y` | `float` | Rendimento dos títulos públicos de 10 anos do governo americano (Treasury) | `^TNX` |
| `vix` | `float` | Índice VIX (Índice de volatilidade / "termômetro do medo") | `^VIX` |

---

### 6. `datas_eventos_brasil.csv`
Série temporal consolidada das datas de eventos relevantes do mercado financeiro e calendário de divulgações no Brasil.
- **Origem**: Expectativas Focus, Decisões do COPOM e regras de vencimento de contratos futuros de Índice Bovespa (WIN).
- **Observação**: Como as três colunas contêm listas de datas com tamanhos e periodicidades diferentes, elas foram alinhadas lado a lado em um único DataFrame, sendo os espaços vazios representados por valores ausentes (`NaN`/vazios).
- **Esquema de Colunas**:

| Nome da Coluna | Tipo de Dado | Descrição |
| :--- | :--- | :--- |
| `data_focus` | `str` (Date) | Datas de publicação de relatórios Focus do Banco Central (geralmente segundas-feiras) |
| `data_copom` | `str` (Date) | Datas de divulgação das decisões de taxas de juros pelo COPOM |
| `data_vencimento_win` | `str` (Date) | Datas oficiais de vencimento do Minicontrato Futuro de Ibovespa (WIN) |

---

### 7. `datas_eventos_g6_eua.csv`
Datas históricas diárias de eventos macroeconômicos importantes nos Estados Unidos.
- **Origem**: FRED (Federal Reserve Economic Data).
- **Esquema de Colunas**:

| Nome da Coluna | Tipo de Dado | Descrição |
| :--- | :--- | :--- |
| `data` | `str` (Date) | Data em que ocorreu a divulgação/evento no formato `YYYY-MM-DD` |
| `evento` | `str` | Tipo do evento macroeconômico. Valores possíveis: `CPI_EUA`, `Juros_Fed`, `PIB_EUA`, `Payroll_NFP` |

---

### 8. `datas_eventos_restantes_g6.csv`
Datas históricas diárias de eventos macroeconômicos importantes do G6 e de relevância global.
- **Origem**: FRED & Dados Locais.
- **Esquema de Colunas**:

| Nome da Coluna | Tipo de Dado | Descrição |
| :--- | :--- | :--- |
| `data` | `str` (Date) | Data em que ocorreu a divulgação/evento no formato `YYYY-MM-DD` |
| `evento` | `str` | Tipo do evento macroeconômico. Valores possíveis: `CPI_EUA`, `IPCA_Brasil`, `Juros_Fed`, `PIB_EUA`, `Payroll_NFP` |

---

## 🐍 Exemplo de Uso Rápido com Python & Pandas

Como todos os CSVs estão no padrão correto, carregá-los no Python é incrivelmente simples e não requer tratamento adicional na leitura:

```python
import pandas as pd

# Exemplo 1: Carregando um arquivo de mercado diário
df_commodities = pd.read_csv('data/dados_macro_commodities.csv', parse_dates=['data'])
print(df_commodities.head())
print(df_commodities.dtypes)

# Exemplo 2: Carregando a Selic
df_selic = pd.read_csv('data/selic.csv', parse_dates=['data'])
print(df_selic.head())

# Exemplo 3: Unindo Ativos Globais com a Selic usando Join pela data
# Ambas as datas já estão no formato ISO YYYY-MM-DD!
df_merged = pd.merge(df_commodities, df_selic, on='data', how='left')
print(df_merged.head())
```
