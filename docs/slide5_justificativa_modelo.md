# Slide 5: Modelagem, Validação e Previsão (Otávio)

## Versão curta para colar no slide

**Validação:** treino de 01/2003 a 09/2022 e teste nos últimos 12 meses (10/2022 a 09/2023)

| Modelo | MAPE |
|---|---|
| **Holt-Winters Multiplicativo** | **2,75%** |
| Holt-Winters Aditivo | 6,58% |
| SARIMA (auto_arima) | 8,46% |

**Modelo escolhido: Holt-Winters Multiplicativo**, com o menor erro no teste. A sazonalidade da série é proporcional ao volume, e o modelo multiplicativo acompanhou melhor a retomada pós-COVID.

**Previsão de 10/2023 a 09/2025:** média de ~81,6M de passageiros/mês, com pico em julho (~92M) e vale em fevereiro (~71M), mais um intervalo de confiança de 95%.

---

## Versão completa (para a fala ou para dúvidas do professor)

## Metodologia
- **Série:** total mensal de passageiros (`Pax`), de 01/2003 a 09/2023, com 249 meses e nenhum mês faltando.
- **Divisão cronológica:** o treino vai de 01/2003 a 09/2022 (237 meses) e o teste cobre os últimos 12 meses, de 10/2022 a 09/2023.
- **Modelos testados:** Holt-Winters aditivo, Holt-Winters multiplicativo (ambos com tendência amortecida e sazonalidade de 12 meses) e SARIMA escolhido pelo `auto_arima`.
- **Métrica:** MAPE (Erro Percentual Absoluto Médio) calculado no conjunto de teste.

## Resultado (Gráfico 3)

| Modelo | MAPE no teste |
|---|---|
| **Holt-Winters Multiplicativo** | **2,75%** |
| Holt-Winters Aditivo | 6,58% |
| SARIMA(3,0,3)x(1,0,1,12) | 8,46% |

## Justificativa da escolha
O Holt-Winters multiplicativo teve o menor erro: em média, a previsão errou 2,75% em relação ao valor real de cada mês do teste. Ele venceu porque, nesta série, a amplitude da sazonalidade acompanha o nível. Quando o volume de passageiros cai, como em 2020, o pico de julho e o vale de janeiro diminuem na mesma proporção. O modelo multiplicativo trata a sazonalidade como percentual do nível e por isso se ajustou melhor à retomada pós-COVID. O SARIMA ficou abaixo do esperado no teste e subestimou o volume da recuperação de 2023, porque a queda abrupta de 2020 distorce os parâmetros autorregressivos.

## Previsão (Gráfico 4)
O modelo vencedor foi treinado de novo com a série completa e projetou os 24 meses de 10/2023 a 09/2025:
- O pico previsto fica em julho, com cerca de **92 milhões** de passageiros, e o vale em fevereiro, com cerca de **71 milhões**.
- A média prevista é de cerca de **81,6 milhões por mês**, volume equivalente ao patamar pré-pandemia de 2019.
- O intervalo de 95% foi calculado a partir do erro relativo do modelo fora do período da COVID (desvio de 2,74%) e se alarga com o horizonte da previsão.

**Limitação:** a tendência amortecida deixa a previsão praticamente estável de um ano para o outro, sem crescimento. Por isso ela é mais confiável no horizonte de 12 meses do que no de 24.
