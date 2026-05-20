# 🎯 Classificação Binária B3 - Ingestão e Processamento de Dados Macro e Commodities

Este repositório contém o pipeline de coleta, padronização e estruturação de dados macroeconômicos, eventos de calendário e commodities globais de alta relevância para modelagem preditiva e classificação binária na B3.

O projeto foi totalmente reorganizado e padronizado para garantir máxima robustez, facilidade de uso em ciência de dados e consistência técnica.

---

## 📂 Estrutura Reorganizada do Projeto

A raiz do projeto foi organizada em diretórios coerentes e modulares:

```
classificacao_binaria_b3/
├── data/                       # Todos os dados limpos, padronizados e validados
│   ├── backup_originais/       # Cópias intocadas dos CSVs brutos originais (segurança)
│   ├── dados_macro_commodities.csv
│   ├── datas_eventos_brasil.csv
│   ├── datas_eventos_g6_eua.csv
│   ├── datas_eventos_restantes_g6.csv
│   ├── ipca.csv
│   ├── minerio_ferro_futuros.csv
│   ├── pib.csv
│   └── selic.csv
├── docs/                       # Documentações de referência do projeto
│   ├── dicionario_dados.md     # GUIA COMPLETO contendo explicações de cada coluna e arquivo
│   └── WIN_Plano_Mestre_v1.pdf # Plano mestre original do projeto (PDF)
├── scripts/                    # Scripts Python modulares para download e atualização de dados
│   ├── baixar_eventos_brasil.py     # Coleta Focus, COPOM e regras WIN do Banco Central do Brasil
│   ├── baixar_eventos_eua.py        # Coleta eventos macro dos EUA via API do FRED
│   └── baixar_macro_commodities.py   # Baixa cotações de ativos e commodities via Yahoo Finance
├── .env                        # Chaves de API locais (FRED_API_KEY)
├── .python-version
├── pyproject.toml              # Definições de dependências do projeto (gerenciado via UV)
├── uv.lock
├── main.py                     # Script principal integrador
└── README.md                   # Este arquivo com as instruções do projeto
```

---

## ⚡ Regras de Padronização de Dados Aplicadas

Para simplificar a análise exploratória e treinamento de modelos de Machine Learning, todos os arquivos na pasta `data/` foram submetidos a um tratamento rigoroso:
1. **Formato Universal**: Delimitador `,` (vírgula), codificação `UTF-8` e separador decimal `.` (ponto).
2. **Coluna Temporal Única**: A coluna de data foi padronizada sob o nome **`data`** em formato ISO **`YYYY-MM-DD`** (séries anuais foram indexadas no dia 1º de janeiro e mensais no dia 1º de cada mês).
3. **Cabeçalhos Limpos**: Todos em formato `snake_case` (letras minúsculas, sem acentos, sem caracteres especiais).
4. **Limpeza Numérica Completa**: 
   - Remoção de sufixos de texto (ex: `0,03K` em volume convertido para float `30.0`).
   - Remoção de símbolos de porcentagem (ex: `-0,19%` salvo como `-0.19`).
   - Remoção de pontos de milhares e vírgulas (ex: `1.796.167,58` convertido para float `1796167.58`).

> [!TIP]
> Para obter uma descrição detalhada de cada campo de cada arquivo, consulte o [Guia de Dicionário de Dados](docs/dicionario_dados.md).

---

## 🚀 Como Executar e Atualizar os Dados

Este repositório utiliza o gerenciador de pacotes **`uv`** para um ambiente virtual extremamente veloz e estável.

### 1. Inicializar o Ambiente
Certifique-se de ter o `uv` instalado em sua máquina. Então, basta executar na pasta raiz:
```bash
uv venv
```

### 2. Configurar Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto contendo a sua credencial da API do FRED:
```env
FRED_API_KEY=sua_chave_aqui
```

### 3. Rodar os Scripts de Atualização de Dados
Para atualizar as tabelas do projeto de forma automatizada, rode os scripts correspondentes a partir da raiz utilizando o Python do ambiente virtual:

* **Baixar Ativos Globais & Commodities (Yahoo Finance)**:
  ```bash
  .venv/bin/python scripts/baixar_macro_commodities.py
  ```

* **Baixar Calendário de Eventos do Mercado Brasileiro (Banco Central)**:
  ```bash
  .venv/bin/python scripts/baixar_eventos_brasil.py
  ```

* **Baixar Calendário de Eventos Americano (FRED API)**:
  ```bash
  .venv/bin/python scripts/baixar_eventos_eua.py
  ```

---

## 🐍 Exemplo Rápido de Carregamento com Pandas

```python
import pandas as pd

# Carregar dados macro e commodities
df_macro = pd.read_csv('data/dados_macro_commodities.csv', parse_dates=['data'])

# Carregar dados da Selic Meta Copom
df_selic = pd.read_csv('data/selic.csv', parse_dates=['data'])

# Mesclar tabelas em segundos usando a chave 'data' padronizada!
df_completo = pd.merge(df_macro, df_selic, on='data', how='left')
print(df_completo.head())
```

---

## 🛡️ Backup e Segurança

Os dados brutos e originais sem padronização foram preservados intactos na pasta [data/backup_originais/](data/backup_originais/) caso seja necessário auditar ou reverter alguma lógica de parse no futuro.
