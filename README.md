# Projeto 2 - C11 Séries Temporais (Grupo 17)

Gabriel Machado Pivoto, Otávio Augusto Silva Lima e Hugo Henrique Reis Camilo

**Dataset:** [U.S. Airline Traffic Data (Kaggle)](https://www.kaggle.com/datasets/yyxian/u-s-airline-traffic-data), com o total mensal de passageiros aéreos nos EUA de 01/2003 a 09/2023. O arquivo está em `data/raw/air traffic.csv`.

## Como rodar

```bash
python3 -m venv .venv
source .venv/bin/activate          # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/modelagem_previsao.py
```

O script leva menos de 1 minuto e gera todos os arquivos de `outputs/`. Não é preciso rodar para fazer os slides, porque os gráficos e tabelas já estão no repositório.

Se aparecer o erro `numpy.dtype size changed`, rode `pip install "numpy<2"`, porque o pmdarima não funciona com o numpy 2.

## O que vai em cada slide

| Slide | Arquivo | Responsável |
|---|---|---|
| 3 | Gráfico 1: série histórica | Gabriel |
| 4 | Gráfico 2: decomposição e respostas sobre tendência, sazonalidade e ciclos | Gabriel |
| 5 | `outputs/figures/grafico4_historico_previsao.png` (gráfico principal) | Otávio (pronto) |
| 5 | `outputs/figures/grafico3_validacao_modelos.png` (apoio, se couber) | Otávio (pronto) |
| 5 | Texto: `docs/slide5_justificativa_modelo.md` (versão curta no topo) | Otávio (pronto) |
| 6 | Conclusões e insights de negócio | Hugo |

## Estrutura

```
data/raw/air traffic.csv         dataset original
src/modelagem_previsao.py         treino/teste, Holt-Winters, SARIMA, MAPE e previsão de 24 meses
outputs/figures/                  gráficos em 300 dpi
outputs/metricas_modelos.csv      MAPE de cada modelo
outputs/previsao_24_meses.csv     previsão mês a mês com intervalo de 95%
docs/                             textos prontos para os slides
```
