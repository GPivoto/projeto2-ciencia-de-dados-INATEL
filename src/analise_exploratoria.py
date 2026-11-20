"""
Projeto 2 (C11 - Series Temporais) - Grupo 17
Parte: Leitura/Tratamento dos Dados, Grafico 1 (serie historica) e
       Grafico 2 (decomposicao) para responder ao Slide 4

Etapas:
  1. Carrega o CSV e trata os dados (separador de milhar, indice datetime, frequencia mensal)
  2. Checagens de qualidade: nulos, duplicados, meses faltando
  3. Grafico 1: serie temporal historica completa (Slide 3)
  4. Decomposicao multiplicativa com period=12 (Grafico 2) e estatisticas de apoio ao Slide 4
  5. Imprime os numeros usados nas respostas de tendencia, sazonalidade e ciclos

Execucao (a partir da raiz do projeto):
    python src/analise_exploratoria.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "air traffic.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SEASON = 12
AZUL, VERMELHO, CINZA = "#0072C6", "#d62728", "#555555"  # azul proximo ao do template Inatel

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
milhoes = mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M")


# 1. Carregamento e tratamento ----------------------------------------------
# Os numeros vem com virgula como separador de milhar ("43,032,450"),
# entao thousands="," faz o pandas ler como inteiro.
df = pd.read_csv(DATA, thousands=",")
df.index = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
df = df.sort_index()

# 2. Checagens de qualidade -------------------------------------------------
print("=== QUALIDADE DOS DADOS ===")
print(f"Linhas: {len(df)} | Colunas: {df.shape[1]}")
print(f"Valores nulos: {int(df.isna().sum().sum())}")
print(f"Meses duplicados: {int(df.index.duplicated().sum())}")

serie = df["Pax"].asfreq("MS")  # frequencia mensal (inicio do mes)
assert serie.isna().sum() == 0, "Serie com meses faltando"
print(f"Periodo: {serie.index.min():%m/%Y} a {serie.index.max():%m/%Y} ({len(serie)} meses, sem lacunas)")
print(f"Minimo: {serie.min() / 1e6:.1f}M em {serie.idxmin():%m/%Y} | "
      f"Maximo: {serie.max() / 1e6:.1f}M em {serie.idxmax():%m/%Y}")

# 3. Grafico 1 - Serie temporal historica (Slide 3) --------------------------
fig, ax = plt.subplots(figsize=(13.33, 6.2))
ax.axvspan(pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31"), color="#f0b27a", alpha=0.25)
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color="#f1948a", alpha=0.25)
ax.plot(serie, color=AZUL, lw=2.4)

ax.annotate("Crise financeira\n2008-2009", xy=(pd.Timestamp("2009-01-01"), 106e6),
            ha="center", va="top", fontsize=12, color="#a04000")
ax.annotate("Mínimo: 3,0M\n(abr/2020, COVID-19)", xy=(serie.idxmin(), serie.min()),
            xytext=(pd.Timestamp("2016-06-01"), 9e6), fontsize=12, color=VERMELHO, ha="center",
            arrowprops=dict(arrowstyle="->", color=VERMELHO, lw=1.5))
ax.annotate("Máximo: 87,8M\n(jul/2023)", xy=(serie.idxmax(), serie.max()),
            xytext=(pd.Timestamp("2021-06-01"), 100e6), fontsize=12, color=CINZA, ha="center",
            arrowprops=dict(arrowstyle="->", color=CINZA, lw=1.5))
ax.annotate("COVID-19", xy=(pd.Timestamp("2021-03-01"), 14e6), ha="center", va="top",
            fontsize=12, color="#922b21")

ax.set_title("Passageiros Aéreos Mensais nos EUA (01/2003 a 09/2023)", weight="bold", fontsize=17)
ax.set_xlabel("Data", fontsize=14); ax.set_ylabel("Passageiros por mês", fontsize=14)
ax.set_ylim(0, 112e6); ax.yaxis.set_major_formatter(milhoes)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig(FIG / "grafico1_serie_historica.png"); plt.close(fig)

# 4. Decomposicao (Grafico 2) -------------------------------------------------
# Modelo multiplicativo: a amplitude sazonal cresce junto com o nivel da serie e
# e o mesmo tipo de sazonalidade que o Holt-Winters vencedor (multiplicativo) assume.
dec = seasonal_decompose(serie, model="multiplicative", period=SEASON)

fig, axes = plt.subplots(4, 1, figsize=(13.33, 9), sharex=True)
paineis = [("Observada", dec.observed, AZUL), ("Tendência", dec.trend, VERMELHO),
           ("Sazonalidade", dec.seasonal, "#2ca02c"), ("Resíduo", dec.resid, "#7f7f7f")]
for ax, (nome, y, cor) in zip(axes, paineis):
    if nome == "Resíduo":
        ax.plot(y, color=cor, lw=1.2, marker="o", ms=2.5, ls="none"); ax.axhline(1, color="black", lw=0.8)
    else:
        ax.plot(y, color=cor, lw=1.8)
    ax.set_ylabel(nome, fontsize=13, weight="bold")
    ax.yaxis.set_major_formatter(milhoes if nome in ("Observada", "Tendência")
                                 else mticker.FormatStrFormatter("%.2f"))
axes[0].set_title("Gráfico 2 - Decomposição Multiplicativa da Série (period = 12)", weight="bold", fontsize=16)
axes[-1].set_xlabel("Data", fontsize=13)
fig.tight_layout(); fig.savefig(FIG / "grafico2_decomposicao.png"); plt.close(fig)

# 5. Numeros de apoio ao Slide 4 ----------------------------------------------
print("\n=== TENDENCIA ===")
anual = serie.groupby(serie.index.year).sum() / 1e6
print((anual.round(1)).to_string())
cagr = (anual[2019] / anual[2003]) ** (1 / 16) - 1
print(f"Crescimento medio anual 2003-2019: {cagr:.2%} a.a. | 2003->2019: {anual[2019] / anual[2003] - 1:.0%}")

print("\n=== SAZONALIDADE (indice sazonal medio por mes; 1,00 = media do ano) ===")
idx = dec.seasonal.groupby(dec.seasonal.index.month).mean()
meses = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
for m, v in idx.items():
    print(f"{meses[m - 1]}: {v:.3f} ({v - 1:+.1%})")
pre = seasonal_decompose(serie[:"2019-12"], model="multiplicative", period=SEASON)
r, s_ = pre.resid.dropna(), pre.seasonal
forca = max(0, 1 - r.var() / (r + s_.loc[r.index]).var())
print(f"Forca da sazonalidade (pre-COVID, 0 a 1): {forca:.2f}")

print("\n=== CICLOS ===")
print(f"2007: {anual[2007]:.1f}M -> 2009: {anual[2009]:.1f}M ({anual[2009] / anual[2007] - 1:+.1%}); "
      f"volume de 2007 so foi superado em {int(anual[anual > anual[2007]].index[anual[anual > anual[2007]].index > 2009][0])}")
print(f"2019: {anual[2019]:.1f}M -> 2020: {anual[2020]:.1f}M ({anual[2020] / anual[2019] - 1:+.1%})")
print(f"Abr/2020 vs Abr/2019: {serie['2020-04-01'] / serie['2019-04-01'] - 1:+.1%}")
print(f"Jul/2023 vs Jul/2019: {serie['2023-07-01'] / serie['2019-07-01'] - 1:+.1%}")
print(f"2022 vs 2019 (ano): {anual[2022] / anual[2019] - 1:+.1%}")
print(f"\nFiguras salvas em {FIG}")
