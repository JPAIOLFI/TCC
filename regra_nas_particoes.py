# -*- coding: utf-8 -*-
"""
Experimento complementar: descoberta da regra DENTRO de cada particao.

Na tese, a regra foi identificada olhando a base inteira e depois aplicada as
mesmas particoes usadas pelos modelos. Aqui o procedimento de descoberta e
repetido usando apenas os atletas de TREINO de cada particao, e a regra obtida
e aplicada aos atletas de TESTE, que nao participaram da descoberta.

Criterio de descoberta (explicito):
  para cada variavel numerica e cada direcao,
    acima: corte = maior valor da classe negativa no treino;
           aceito se ha >= suporte registros de treino acima do corte e todos sao positivos
    abaixo: corte = menor valor da classe negativa no treino;
           aceito se ha >= suporte registros de treino abaixo do corte e todos sao positivos
  a regra e a disjuncao (OU) dos cortes aceitos.
O experimento e rodado com dois suportes minimos (1 e 5 registros): com 1, um
unico registro extremo que por acaso seja positivo ja cria um corte.

Uso:  python regra_nas_particoes.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parent
RANDOM_STATE, N_SPLITS = 42, 10
TARGET, GROUP = 'Injury_Risk', 'Athlete_ID'
NUMERICAS = ['Heart_Rate', 'Muscle_Activity', 'Oxygen_Saturation', 'Age',
             'Motion_Z', 'Motion_X', 'Motion_Y']


def descobrir(treino, suporte):
    """Devolve a lista de cortes aceitos: (variavel, direcao, valor)."""
    y = treino[TARGET]
    cortes = []
    for c in NUMERICAS:
        neg = treino.loc[y == 0, c]
        for direcao, valor, alem in [('>', neg.max(), treino[c] > neg.max()),
                                     ('<', neg.min(), treino[c] < neg.min())]:
            if alem.sum() >= suporte and y[alem].all():
                cortes.append((c, direcao, valor))
    return cortes


def aplicar(df, cortes):
    pred = np.zeros(len(df), dtype=bool)
    for c, d, v in cortes:
        pred |= (df[c] > v).values if d == '>' else (df[c] < v).values
    return pred.astype(int)


def rodar(arquivo, rotulo, saida, suporte):
    df = pd.read_csv(BASE / 'data' / arquivo)
    y = df[TARGET].astype(int).values
    regra_fixa = ((df.Heart_Rate > 180) | (df.Muscle_Activity > 80)).astype(int).values
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print('=' * 78)
    print(f'{rotulo}: regra descoberta so no treino, suporte minimo de {suporte} registro(s)')
    print('=' * 78)
    linhas = []
    for i, (tr, te) in enumerate(cv.split(df, y, df[GROUP])):
        cortes = descobrir(df.iloc[tr], suporte)
        teste = df.iloc[te]
        pred = aplicar(teste, cortes)
        yt = y[te]
        excecoes = int(((pred == 1) & (yt == 0)).sum())
        texto = ' OU '.join(f'{c} {d} {v:.3f}' for c, d, v in cortes) or '(nenhum corte)'
        linhas.append({
            'particao': i, 'cortes': texto, 'n_cortes': len(cortes),
            'apontados_no_teste': int(pred.sum()), 'excecoes_no_teste': excecoes,
            'precisao': precision_score(yt, pred, zero_division=0),
            'recall': recall_score(yt, pred, zero_division=0),
            'f1': f1_score(yt, pred, zero_division=0),
            'f1_regra_fixa': f1_score(yt, regra_fixa[te], zero_division=0),
            'predicoes_iguais_a_regra_fixa': int((pred == regra_fixa[te]).sum()),
            'registros_no_teste': len(te)})
        l = linhas[-1]
        print(f'  particao {i}: {texto}')
        print(f'     teste: {l["apontados_no_teste"]} apontados, {excecoes} excecoes | '
              f'F1 {l["f1"]:.4f} (regra fixa {l["f1_regra_fixa"]:.4f}) | '
              f'iguais a regra fixa {l["predicoes_iguais_a_regra_fixa"]} de {len(te)}')
    t = pd.DataFrame(linhas)
    print('-' * 78)
    print(f'  F1 medio, regra descoberta no treino : {t.f1.mean():.4f} +- {t.f1.std(ddof=0):.4f}')
    print(f'  F1 medio, regra fixa da base inteira : {t.f1_regra_fixa.mean():.4f}')
    print(f'  precisao media                       : {t.precisao.mean():.4f}')
    print(f'  total de excecoes no teste           : {t.excecoes_no_teste.sum()} '
          f'em {t.apontados_no_teste.sum()} registros apontados')
    freq = pd.Series([x.split(' ')[0] + ' ' + x.split(' ')[1] for c in t.cortes
                      for x in c.split(' OU ') if x != '(nenhum corte)']).value_counts()
    print('  cortes encontrados (em quantas particoes): ' +
          ', '.join(f'{k} em {v}' for k, v in freq.items()))
    print(f'  predicoes identicas a regra fixa     : {t.predicoes_iguais_a_regra_fixa.sum():,} '
          f'de {t.registros_no_teste.sum():,}')
    saida.mkdir(exist_ok=True)
    t.to_csv(saida / f'regra_descoberta_por_particao_suporte{suporte}.csv', index=False,
             float_format='%.6f', encoding='utf-8-sig')
    print(f'  -> {saida.name}/regra_descoberta_por_particao_suporte{suporte}.csv\n')


for suporte in [1, 5]:
    rodar('athlete_physiological_dataset.csv', 'BASE COMPLETA', BASE / 'outputs', suporte)
    rodar('soccer_physiological_data.csv', 'SUBGRUPO DE FUTEBOL', BASE / 'outputs_soccer', suporte)
