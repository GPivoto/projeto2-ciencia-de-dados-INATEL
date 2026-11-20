# Projeto 2 - C11 Séries Temporais (Grupo 17)

Gabriel Machado Pivoto, Otávio Augusto Silva Lima e Hugo Henrique Reis Camilo

**Dataset:** [U.S. Airline Traffic Data (Kaggle)](https://www.kaggle.com/datasets/yyxian/u-s-airline-traffic-data), com o total mensal de passageiros aéreos nos EUA de 01/2003 a 09/2023. O arquivo está em `data/raw/air traffic.csv`.

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate          # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/analise_exploratoria.py  
python src/modelagem_previsao.py     
```

O script leva menos de 1 minuto e gera todos os arquivos de `outputs/`. Não é preciso rodar para fazer os slides, porque os gráficos e tabelas já estão no repositório.

Se aparecer o erro `numpy.dtype size changed`, rode `pip install "numpy<2"`, porque o pmdarima não funciona com o numpy 2.

## Estrutura

```
data/raw/air traffic.csv         dataset original
src/analise_exploratoria.py       tratamento dos dados, Gráfico 1 e decomposição (Gráfico 2)
src/modelagem_previsao.py         treino/teste, Holt-Winters, SARIMA, MAPE e previsão de 24 meses
outputs/figures/                  gráficos em 300 dpi
outputs/metricas_modelos.csv      MAPE de cada modelo
outputs/previsao_24_meses.csv     previsão mês a mês com intervalo de 95%

```
