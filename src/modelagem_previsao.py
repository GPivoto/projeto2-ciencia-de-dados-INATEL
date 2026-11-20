"""
Projeto 2 (C11 - Series Temporais) - Grupo 17
Parte: Modelagem Estatistica, Avaliacao de Erro (MAPE) e Previsao Futura

Etapas:
  1. Carrega a serie mensal de passageiros (Pax) do U.S. Airline Traffic Data
  2. Divide cronologicamente em Treino e Teste (ultimos 12 meses = teste)
  3. Treina Holt-Winters (aditivo e multiplicativo) e SARIMA (auto_arima)
  4. Calcula o MAPE no teste e escolhe o modelo vencedor
  5. Grafico 3: Treino vs Teste vs Previsao dos modelos
  6. Re-treina o vencedor na serie completa e preve os proximos 24 meses
  7. Grafico 4: Serie historica + previsao futura (com intervalo de 95%)

Execucao (a partir da raiz do projeto):
    python src/modelagem_previsao.py
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "air traffic.csv"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TEST_MONTHS = 12
HORIZON = 24
SEASON = 12

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
milhoes = mticker.FuncFormatter(lambda x, _: f"{x / 1e6:.0f}M")


def mape(real, prev):
    real, prev = np.asarray(real), np.asarray(prev)
    return float(np.mean(np.abs((real - prev) / real)) * 100)


# 1. Carregamento -----------------------------------------------------------
df = pd.read_csv(DATA, thousands=",")
df.index = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
serie = df["Pax"].sort_index().asfreq("MS")
assert serie.isna().sum() == 0, "Serie com meses faltando"
print(f"Serie: {serie.index.min():%Y-%m} a {serie.index.max():%Y-%m} ({len(serie)} meses)")

# 2. Treino / Teste cronologico ---------------------------------------------
treino, teste = serie[:-TEST_MONTHS], serie[-TEST_MONTHS:]
print(f"Treino: {treino.index.min():%Y-%m} a {treino.index.max():%Y-%m} ({len(treino)})")
print(f"Teste : {teste.index.min():%Y-%m} a {teste.index.max():%Y-%m} ({len(teste)})")

# 3. Modelos ----------------------------------------------------------------
def holt_winters(y, sazonal):
    return ExponentialSmoothing(y, trend="add", seasonal=sazonal, damped_trend=True,
                                seasonal_periods=SEASON).fit(optimized=True)


modelos = {
    "Holt-Winters Aditivo": holt_winters(treino, "add"),
    "Holt-Winters Multiplicativo": holt_winters(treino, "mul"),
}
sarima = pm.auto_arima(treino, seasonal=True, m=SEASON, stepwise=True,
                       suppress_warnings=True, error_action="ignore")
nome_sarima = f"SARIMA{sarima.order}x{sarima.seasonal_order}"

previsoes = {n: m.forecast(TEST_MONTHS) for n, m in modelos.items()}
previsoes[nome_sarima] = pd.Series(sarima.predict(TEST_MONTHS), index=teste.index)

# 4. MAPE -------------------------------------------------------------------
metricas = (pd.DataFrame({"Modelo": list(previsoes),
                          "MAPE (%)": [mape(teste, p) for p in previsoes.values()]})
            .sort_values("MAPE (%)").reset_index(drop=True))
metricas["MAPE (%)"] = metricas["MAPE (%)"].round(2)
metricas.to_csv(OUT / "metricas_modelos.csv", index=False)
vencedor = metricas.loc[0, "Modelo"]
print("\nMAPE no conjunto de teste:\n" + metricas.to_string(index=False))
print(f"\nModelo vencedor: {vencedor}")

# 5. Grafico 3 - Validacao --------------------------------------------------
cores = ["#d62728", "#2ca02c", "#9467bd"]
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(treino["2018":], color="#1f77b4", lw=2, label="Treino")
ax.plot(teste, color="black", lw=2.5, label="Teste (real)")
for (nome, prev), cor in zip(previsoes.items(), cores):
    erro = metricas.set_index("Modelo").loc[nome, "MAPE (%)"]
    destaque = nome == vencedor
    ax.plot(prev, color=cor, lw=2.5 if destaque else 1.6, ls="-" if destaque else "--",
            label=f"{nome}  |  MAPE = {erro:.2f}%" + ("  (vencedor)" if destaque else ""))
ax.axvline(teste.index[0], color="gray", ls=":", lw=1)
ax.set_title("Gráfico 3 - Validação dos Modelos: Treino vs Teste vs Previsão", weight="bold")
ax.set_xlabel("Data"); ax.set_ylabel("Passageiros por mês")
ax.yaxis.set_major_formatter(milhoes); ax.legend(loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(FIG / "grafico3_validacao_modelos.png"); plt.close(fig)

# 6. Re-treino do vencedor na serie completa + previsao futura ---------------
datas_fut = pd.date_range(serie.index[-1] + pd.offsets.MonthBegin(), periods=HORIZON, freq="MS")
if vencedor.startswith("Holt-Winters"):
    final = holt_winters(serie, "mul" if "Multiplicativo" in vencedor else "add")
    futuro = final.forecast(HORIZON)
    # Intervalo de 95% a partir do erro relativo in-sample, excluindo o choque da
    # COVID (03/2020 a 12/2021), que inflaria artificialmente a variancia.
    # A incerteza cresce com a raiz do horizonte (h).
    erro_rel = (serie - final.fittedvalues) / final.fittedvalues
    covid = (erro_rel.index >= "2020-03-01") & (erro_rel.index <= "2021-12-01")
    sigma = erro_rel[~covid].std()
    h = np.sqrt(np.arange(1, HORIZON + 1))
    lim_inf, lim_sup = futuro * (1 - 1.96 * sigma * h), futuro * (1 + 1.96 * sigma * h)
    print(f"Desvio do erro relativo (sem COVID): {sigma:.2%}")
else:
    final = pm.auto_arima(serie, seasonal=True, m=SEASON, stepwise=True, suppress_warnings=True)
    fc, ci = final.predict(HORIZON, return_conf_int=True)
    futuro = pd.Series(fc, index=datas_fut)
    lim_inf, lim_sup = pd.Series(ci[:, 0], index=datas_fut), pd.Series(ci[:, 1], index=datas_fut)

prev_df = pd.DataFrame({"Previsao_Pax": futuro.round(0), "IC95_inf": lim_inf.round(0),
                        "IC95_sup": lim_sup.round(0)}, index=datas_fut)
prev_df.index.name = "Data"
prev_df.to_csv(OUT / "previsao_24_meses.csv")
print(f"\nPrevisao {datas_fut[0]:%Y-%m} a {datas_fut[-1]:%Y-%m}:")
print(prev_df.applymap(lambda v: f"{v / 1e6:,.1f}M").to_string())

# 7. Grafico 4 - Historico + Previsao ---------------------------------------
fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(serie, color="#1f77b4", lw=1.6, label="Histórico (2003-2023)")
ax.plot(pd.concat([serie[-1:], futuro]), color="#d62728", lw=2.2,
        label=f"Previsão {HORIZON} meses ({vencedor})")
ax.fill_between(datas_fut, lim_inf, lim_sup, color="#d62728", alpha=0.15, label="Intervalo de 95%")
ax.axvline(serie.index[-1], color="gray", ls=":", lw=1)
ax.set_title(f"Gráfico 4 - Passageiros Aéreos nos EUA: Série Histórica + Previsão "
             f"({datas_fut[0]:%m/%Y} a {datas_fut[-1]:%m/%Y})", weight="bold")
ax.set_xlabel("Data"); ax.set_ylabel("Passageiros por mês")
ax.yaxis.set_major_formatter(milhoes); ax.legend(loc="lower left", frameon=False)
fig.tight_layout(); fig.savefig(FIG / "grafico4_historico_previsao.png"); plt.close(fig)

print(f"\nArquivos salvos em {OUT}")
