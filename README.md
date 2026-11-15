# Rotulagem e protocolos de validação na avaliação de classificadores em bases sintéticas de monitoramento esportivo

Código e dados do Trabalho de Conclusão de Curso de João Pedro Aiolfi de Figueiredo
(Bacharelado em Ciência da Computação, PUCRS, 2026), sob orientação da
Profa. Dra. Andréa Aparecida Konzen.

O trabalho examina uma base pública e sintética de monitoramento de atletas e mostra que:

1. os rótulos de risco são compatíveis com uma regra simples
   (`Heart_Rate > 180` ou `Muscle_Activity > 80`) mais uma componente sem sinal detectável;
2. o melhor classificador reproduz essa regra registro a registro, e nenhum modelo testado a supera;
3. um protocolo inadequado de reamostragem (SMOTE antes da divisão dos dados, sem agrupamento
   por atleta) faz o recall subir de 0,11 para 0,95 sobre a mesma base.

## Como reproduzir

Requer Python 3.13.

```bash
pip install -r requirements.txt
python pipeline.py     # todos os números do Capítulo 5, em outputs/ e outputs_soccer/ (15 a 25 min)
python figuras.py      # todas as figuras, em fig_leve/ (lê os resultados do passo anterior)
python regra_nas_particoes.py   # experimento da Seção 5.2.3 (alguns segundos)
```

Todas as operações aleatórias usam a semente 42. Os resultados são determinísticos com as
versões listadas em `requirements.txt`.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `pipeline.py` | Experimento completo, organizado na ordem do Capítulo 5 |
| `figuras.py` | Figuras da tese, geradas a partir de `outputs/` |
| `regra_nas_particoes.py` | Descoberta da regra usando só os atletas de treino de cada partição |
| `data/athlete_physiological_dataset.csv` | Base principal: 23.400 registros de 234 atletas |
| `data/soccer_physiological_data.csv` | Recorte de futebol: 5.100 registros, todos contidos na base principal |
| `outputs/` | Resultados da base completa (CSV) |
| `outputs_soccer/` | Resultados do subgrupo de futebol (CSV) |
| `fig_leve/` | Figuras usadas no texto |
| `tcc_latex.tex` | Texto da monografia (classe `tcc` da PUCRS) |

## Onde está cada resultado da tese

| Tese | Arquivo em `outputs/` |
|---|---|
| Tabela 5.1, d de Cohen | `tamanho_efeito_features.csv` |
| Seção 5.1.3, regra e resíduo | `resumo.csv` |
| Tabela 5.2, Random Forest só no resíduo | `residuo_random_forest.csv` |
| Tabela 5.3, regra mais *k* registros do resíduo | `varredura_k.csv` |
| Tabela 5.4, 11 configurações | `resultados_comparativos.csv` |
| Tabela 5.5, regra por partição | `regra_por_particao.csv` |
| Seção 5.2.2, coincidência de 23.400 registros | `predicoes_por_particao.csv` |
| Tabela 5.6, regra descoberta no treino de cada partição | `regra_descoberta_por_particao_suporte5.csv` (e `_suporte1.csv`) |
| Tabelas 5.7 e 5.8, *boosting* e busca aninhada | `boosting_e_tuning.csv`, `busca_aninhada_particoes.csv`, `busca_aninhada_escolhas.csv` |
| Tabela 5.9, ajuste de limiar | `ajuste_limiar.csv` |
| Tabela 5.10, explicabilidade | `ranking_explicabilidade.csv` |
| Figura 5.9, árvore ilustrativa | `arvore_folhas.csv` |
| Tabelas 5.11 e 5.12, protocolos de SMOTE | `protocolos_smote.csv`, `leakage_smote.csv` |
| Tabela 5.13, agrupamento por atleta | `leakage_grupo.csv` |
| Tabela 5.14, subgrupo de futebol | arquivos de mesmo nome em `outputs_soccer/` |

O arquivo `predicoes_por_particao.csv` traz, para cada um dos 23.400 registros, o atleta, a
partição em que foi avaliado, o rótulo, a saída da regra e a predição fora da amostra do
Random Forest. Ele permite conferir diretamente que as duas predições coincidem em todos os registros.

## Decisões de protocolo

- **Validação:** `StratifiedGroupKFold` com 10 partições, agrupado por `Athlete_ID`. Nenhum atleta
  aparece em treino e teste ao mesmo tempo.
- **Pré-processamento:** o *one-hot* das variáveis categóricas é feito na base inteira, porque não
  usa o alvo nem estatísticas estimadas. O `StandardScaler` é ajustado só no treino de cada
  partição e apenas transforma o teste. O SMOTE, quando usado, reamostra só o treino.
- **Desvio-padrão:** calculado entre as 10 partições com divisão por *n* (`numpy.std`).
- **`scale_pos_weight` do XGBoost:** razão entre negativos e positivos, calculada uma vez na base inteira.
- **Busca de hiperparâmetros:** a estimativa principal do Random Forest otimizado vem da validação
  cruzada aninhada. A busca única sobre a base inteira é mantida só para comparação.
- **Precisão média (AP):** calculada por `average_precision_score`, e não por integração trapezoidal.

## Dados

Base pública *Real-Time Physiological Monitoring in Sports: Sports Training Injury Warning Data*,
publicada no Kaggle pelo usuário `colabsss` sob licença CC0 1.0 (domínio público) e obtida em
11 de maio de 2026:
<https://www.kaggle.com/datasets/colabsss/real-time-physiological-monitoring-in-sports>.
Os dados são simulados e não correspondem a atletas reais.
