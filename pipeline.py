# -*- coding: utf-8 -*-
"""
Rotulagem e protocolos de validacao na avaliacao de classificadores em bases
sinteticas de monitoramento esportivo. TCC II, Ciencia da Computacao, PUCRS.

Este script reproduz todos os numeros do Capitulo 5. Cada secao indica a
tabela da tese que ela gera. Os resultados sao gravados em outputs/ (base
completa) e outputs_soccer/ (subgrupo de futebol). As figuras sao geradas a
partir desses arquivos por figuras.py.

Uso:  python pipeline.py          (cerca de 15 a 25 minutos)
"""
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold, GroupShuffleSplit,
                                     cross_validate, cross_val_predict, RandomizedSearchCV)
from sklearn.metrics import (make_scorer, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, precision_recall_curve)
from sklearn.inspection import permutation_importance
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# =============================================================================
# 0. CONFIGURACAO
# =============================================================================
BASE = Path(__file__).resolve().parent
ARQ_PRINCIPAL = BASE / 'data' / 'athlete_physiological_dataset.csv'
ARQ_SOCCER = BASE / 'data' / 'soccer_physiological_data.csv'
RANDOM_STATE, N_SPLITS = 42, 10
TARGET, GROUP = 'Injury_Risk', 'Athlete_ID'
NUMERICAS = ['Heart_Rate', 'Muscle_Activity', 'Oxygen_Saturation', 'Age',
             'Motion_Z', 'Motion_X', 'Motion_Y']

SCORING = {'accuracy': 'accuracy',
           'precision': make_scorer(precision_score, zero_division=0),
           'recall': make_scorer(recall_score, zero_division=0),
           'f1': make_scorer(f1_score, zero_division=0),
           'roc_auc': 'roc_auc',
           'avg_prec': 'average_precision'}          # precisao media (AP)

GRADE_RF = {'clf__n_estimators': [100, 200, 400],
            'clf__max_depth': [None, 5, 10, 20],
            'clf__min_samples_split': [2, 5, 10, 20],
            'clf__min_samples_leaf': [1, 2, 4, 8],
            'clf__max_features': ['sqrt', 'log2', None],
            'clf__class_weight': [None, 'balanced']}

T0 = time.time()


def titulo(txt):
    print('\n' + '=' * 76 + f'\n{txt}   [{time.time() - T0:.0f}s]\n' + '=' * 76)


def preparar(df):
    """X, y e grupos. O one-hot e feito na base inteira: nao usa o alvo nem
    estatisticas estimadas, portanto nao introduz vazamento."""
    y = df[TARGET].astype(int)
    g = df[GROUP]
    X = df.drop(columns=[TARGET, GROUP])
    X = pd.get_dummies(X, columns=X.select_dtypes('object').columns.tolist(),
                       drop_first=True, dtype=float)
    return X, y, g


def pipe(clf):
    """StandardScaler ajustado so na porcao de treino de cada particao."""
    return Pipeline([('scaler', StandardScaler()), ('clf', clf)])


def pipe_smote(clf):
    """SMOTE dentro do pipeline: roda no fit (treino) e e ignorado no predict."""
    return ImbPipeline([('scaler', StandardScaler()),
                        ('smote', SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
                        ('clf', clf)])


def rf(**kw):
    return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1, **kw)


def cv_agrupado(k=N_SPLITS):
    return StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)


def avaliar(modelo, X, y, cv, grupos=None, n_jobs=-1):
    """Media e desvio-padrao (populacional, ddof=0) das metricas entre particoes."""
    r = cross_validate(modelo, X, y, cv=cv, groups=grupos, scoring=SCORING,
                       n_jobs=n_jobs, error_score='raise')
    linha = {}
    for m in SCORING:
        linha[m] = r[f'test_{m}'].mean()
        linha[f'{m}_std'] = r[f'test_{m}'].std()
    return linha


def cohen_d(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (b.mean() - a.mean()) / s if s > 0 else 0.0


def salvar(df, pasta, nome, **kw):
    pasta.mkdir(exist_ok=True)
    df.to_csv(pasta / nome, float_format='%.6f', encoding='utf-8-sig', **kw)
    print(f'  -> {pasta.name}/{nome}')


# =============================================================================
# EXPERIMENTO COMPLETO SOBRE UM CONJUNTO DE DADOS
# =============================================================================
def experimento(df, rotulo, saida, completo=True):
    X, y, g = preparar(df)
    cv = cv_agrupado()
    R = {}

    # -------------------------------------------------------------------------
    titulo(f'[{rotulo}] 5.1.1 e 5.1.2  Distribuicao do alvo e d de Cohen (Tabela 5.1)')
    print(f'  registros {len(df):,} | atletas {g.nunique()} | positivos {int(y.sum()):,} '
          f'({y.mean():.4f})')
    g0, g1 = df[y == 0], df[y == 1]
    efeitos = pd.DataFrame({'media_classe_0': [g0[c].mean() for c in NUMERICAS],
                            'media_classe_1': [g1[c].mean() for c in NUMERICAS],
                            'd_cohen': [cohen_d(g0[c], g1[c]) for c in NUMERICAS]},
                           index=NUMERICAS)
    efeitos['abs_d'] = efeitos.d_cohen.abs()
    print(efeitos.sort_values('abs_d', ascending=False).round(3).to_string())
    salvar(efeitos.sort_values('abs_d', ascending=False), saida, 'tamanho_efeito_features.csv')

    # -------------------------------------------------------------------------
    titulo(f'[{rotulo}] 5.1.3  Busca da regra: maior valor da classe 0 em cada variavel')
    cortes = {}
    for c in NUMERICAS:
        max0 = df.loc[y == 0, c].max()
        acima = df[df[c] > max0]
        if len(acima) and acima[TARGET].all():
            cortes[c] = max0
            print(f'  {c:<18} > {max0:.4f}: {len(acima)} registros, todos positivos')
    na_regra = np.zeros(len(df), dtype=bool)
    for c, t in cortes.items():
        na_regra |= (df[c] > t).values
    # O texto usa os cortes abreviados 180 e 80; confere que dao o mesmo resultado
    regra_abrev = ((df.Heart_Rate > 180) | (df.Muscle_Activity > 80)).values
    print(f'  cortes exatos e abreviados (180 e 80) coincidem em todos os registros: '
          f'{bool((na_regra == regra_abrev).all())}')
    pos_regra, pos_total = int(y[na_regra].sum()), int(y.sum())
    resid = df[~na_regra]
    taxa_resid = resid[TARGET].mean()
    ef_resid = pd.Series({c: abs(cohen_d(resid.loc[resid[TARGET] == 0, c],
                                         resid.loc[resid[TARGET] == 1, c])) for c in NUMERICAS})
    rec_regra = pos_regra / pos_total
    f1_base_inteira = 2 * rec_regra / (1 + rec_regra)
    print(f'  registros na regra {int(na_regra.sum())} | positivos neles {pos_regra} | '
          f'cobertura {rec_regra:.4f}')
    print(f'  taxa de positivos fora da regra {taxa_resid:.4f} | maior |d| no residuo '
          f'{ef_resid.max():.3f} ({ef_resid.idxmax()})')
    print(f'  F1 da regra na base inteira {f1_base_inteira:.4f}')
    R.update(n_regra=int(na_regra.sum()), cobertura=rec_regra, taxa_resid=taxa_resid,
             maior_d_resid=ef_resid.max(), f1_regra_base=f1_base_inteira)

    if completo:
        # ---------------------------------------------------------------------
        titulo(f'[{rotulo}] 5.1.4  Random Forest so no residuo (Tabela 5.2)')
        Xr, yr, gr = X[~na_regra], y[~na_regra], g[~na_regra]
        r = cross_validate(pipe(rf()), Xr, yr, cv=cv_agrupado(), groups=gr, n_jobs=-1,
                           scoring=['precision', 'recall', 'f1', 'roc_auc', 'average_precision'])
        tab = pd.DataFrame({k: [r[f'test_{k}'].mean(), r[f'test_{k}'].std()]
                            for k in ['precision', 'recall', 'f1', 'roc_auc', 'average_precision']},
                           index=['media', 'desvio']).T
        tab['referencia_acaso'] = [yr.mean(), np.nan, np.nan, 0.5, yr.mean()]
        print(tab.round(4).to_string())
        salvar(tab, saida, 'residuo_random_forest.csv')

        # ---------------------------------------------------------------------
        titulo(f'[{rotulo}] 5.1.5  Regra + k registros do residuo ao acaso (Tabela 5.3)')
        n_res = int((~na_regra).sum())
        linhas = []
        for k in [0, 100, 500, 1000, 5000, 10000, n_res]:
            tp = pos_regra + taxa_resid * k
            fp = (1 - taxa_resid) * k
            fn = pos_total - tp
            linhas.append({'k': k, 'precisao': tp / (tp + fp), 'recall': tp / pos_total,
                           'f1': 2 * tp / (2 * tp + fp + fn)})
        tab = pd.DataFrame(linhas)
        print(tab.round(4).to_string(index=False))
        salvar(tab, saida, 'varredura_k.csv', index=False)

    # -------------------------------------------------------------------------
    titulo(f'[{rotulo}] 5.2.1  As 11 configuracoes (Tabela 5.4)')
    configs = {
        '[Baseline] Dummy (Majoritário)': pipe(DummyClassifier(strategy='most_frequent')),
        '[Baseline] Dummy (Estratificado)': pipe(DummyClassifier(strategy='stratified',
                                                                 random_state=RANDOM_STATE)),
        '[Baseline] Regressão Logística': pipe(LogisticRegression(max_iter=1000,
                                                                  random_state=RANDOM_STATE)),
        '[Baseline] Árvore de Decisão': pipe(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        '[Baseline] Random Forest': pipe(rf()),
        '[Weights] LR': pipe(LogisticRegression(max_iter=1000, class_weight='balanced',
                                                random_state=RANDOM_STATE)),
        '[Weights] DT': pipe(DecisionTreeClassifier(class_weight='balanced',
                                                    random_state=RANDOM_STATE)),
        '[Weights] RF': pipe(rf(class_weight='balanced')),
        '[SMOTE] LR': pipe_smote(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        '[SMOTE] DT': pipe_smote(DecisionTreeClassifier(random_state=RANDOM_STATE)),
        '[SMOTE] RF': pipe_smote(rf()),
    }
    res = {}
    for nome, m in configs.items():
        res[nome] = avaliar(m, X, y, cv, g)
        print(f'  {nome:<34} F1 {res[nome]["f1"]:.4f} +- {res[nome]["f1_std"]:.4f}  '
              f'ROC-AUC {res[nome]["roc_auc"]:.4f}')
    tab = pd.DataFrame(res).T.sort_values('f1', ascending=False)
    salvar(tab, saida, 'resultados_comparativos.csv')
    R['melhor_f1'] = tab.f1.max()

    # -------------------------------------------------------------------------
    titulo(f'[{rotulo}] 5.2.2  Regra como classificador nas mesmas particoes (Tabela 5.5)')
    regra = na_regra.astype(int)
    linhas = []
    for i, (_, te) in enumerate(cv.split(X, y, g)):
        linhas.append({'particao': i, 'precisao': precision_score(y.iloc[te], regra[te],
                                                                  zero_division=0),
                       'recall': recall_score(y.iloc[te], regra[te]),
                       'f1': f1_score(y.iloc[te], regra[te])})
    rp = pd.DataFrame(linhas)
    R['f1_regra_particoes'] = rp.f1.mean()
    print(f'  regra: precisao {rp.precisao.mean():.4f} | recall {rp.recall.mean():.4f} | '
          f'F1 {rp.f1.mean():.4f}')
    salvar(rp, saida, 'regra_por_particao.csv', index=False)

    # predicoes fora da amostra do Random Forest baseline, identificadas por registro
    pred = cross_val_predict(pipe(rf()), X, y, cv=cv, groups=g, n_jobs=-1)
    particao = np.empty(len(y), dtype=int)
    for i, (_, te) in enumerate(cv.split(X, y, g)):
        particao[te] = i
    pr = pd.DataFrame({'registro': df.index, 'Athlete_ID': g.values, 'particao': particao,
                       'Injury_Risk': y.values, 'regra': regra, 'random_forest': pred})
    iguais = int((pr.regra == pr.random_forest).sum())
    print(f'  Random Forest e regra coincidem em {iguais:,} de {len(pr):,} registros')
    salvar(pr, saida, 'predicoes_por_particao.csv', index=False)

    # -------------------------------------------------------------------------
    titulo(f'[{rotulo}] 5.2.4  Gradient boosting e RF otimizado nao aninhado (Tabela 5.7)')
    spw = (y == 0).sum() / (y == 1).sum()          # calculado uma vez, na base inteira

    def xgb(**kw):
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, eval_metric='logloss',
                             random_state=RANDOM_STATE, n_jobs=1, **kw)
    # n_jobs=1: o XGBoost nao da resultados identicos com numeros diferentes de
    # nucleos (a ordem das somas muda). Um nucleo garante a reproducao exata.

    busca = RandomizedSearchCV(pipe(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
                               GRADE_RF, n_iter=25, scoring='f1', cv=cv_agrupado(3),
                               random_state=RANDOM_STATE, n_jobs=-1)
    busca.fit(X, y, groups=g)
    print(f'  scale_pos_weight = {spw:.3f}')
    print(f'  busca nao aninhada escolheu: '
          f'{ {k.replace("clf__", ""): v for k, v in busca.best_params_.items()} }')
    modelos = {'[Boosting] XGBoost': pipe(xgb()),
               '[Boosting] XGBoost + scale_pos_weight': pipe(xgb(scale_pos_weight=spw)),
               '[Boosting] HistGradientBoosting': pipe(HistGradientBoostingClassifier(
                   max_iter=300, random_state=RANDOM_STATE)),
               '[Tuned] Random Forest (nao aninhada)': busca.best_estimator_}
    rob = {}
    for nome, m in modelos.items():
        # O XGBoost ja paraleliza internamente; rodar as particoes em processos
        # paralelos causa falha de memoria no Windows.
        rob[nome] = avaliar(m, X, y, cv, g, n_jobs=1 if 'XGBoost' in nome else -1)
        print(f'  {nome:<40} F1 {rob[nome]["f1"]:.4f}  ROC-AUC {rob[nome]["roc_auc"]:.4f}')

    # -------------------------------------------------------------------------
    titulo(f'[{rotulo}] 5.2.4  Busca aninhada do Random Forest (Tabelas 5.7 e 5.8)')
    linhas, escolhas = [], []
    for i, (tr, te) in enumerate(cv.split(X, y, g)):
        b = RandomizedSearchCV(pipe(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
                               GRADE_RF, n_iter=25, scoring='f1', cv=cv_agrupado(3),
                               random_state=RANDOM_STATE, n_jobs=-1)
        b.fit(X.iloc[tr], y.iloc[tr], groups=g.iloc[tr])
        p = b.best_estimator_.predict(X.iloc[te])
        pp = b.best_estimator_.predict_proba(X.iloc[te])[:, 1]
        linhas.append({'precision': precision_score(y.iloc[te], p, zero_division=0),
                       'recall': recall_score(y.iloc[te], p), 'f1': f1_score(y.iloc[te], p),
                       'roc_auc': roc_auc_score(y.iloc[te], pp),
                       'avg_prec': average_precision_score(y.iloc[te], pp)})
        escolhas.append({k.replace('clf__', ''): v for k, v in b.best_params_.items()})
        print(f'  particao {i}: F1 {linhas[-1]["f1"]:.4f}  precisao {linhas[-1]["precision"]:.4f}  '
              f'class_weight={escolhas[-1]["class_weight"]}')
    an = pd.DataFrame(linhas)
    salvar(an, saida, 'busca_aninhada_particoes.csv', index_label='particao')
    salvar(pd.DataFrame(escolhas), saida, 'busca_aninhada_escolhas.csv', index_label='particao')
    for chave in ['n_estimators', 'max_depth', 'min_samples_split', 'class_weight']:
        print(f'    {chave:<18} {dict(Counter(str(e[chave]) for e in escolhas))}')
    rob['[Tuned] Random Forest (aninhada)'] = {
        **{m: an[m].mean() for m in an}, **{f'{m}_std': an[m].std(ddof=0) for m in an}}
    print(f'  aninhada: F1 {an.f1.mean():.4f} +- {an.f1.std(ddof=0):.4f}')
    tab = pd.DataFrame(rob).T
    tab['pct_da_regra_nas_particoes'] = 100 * tab.f1 / R['f1_regra_particoes']
    salvar(tab, saida, 'boosting_e_tuning.csv')

    # -------------------------------------------------------------------------
    titulo(f'[{rotulo}] 5.4.1  Protocolos de SMOTE (Tabelas 5.11 e 5.12)')
    Xa, ya = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
    cv_a = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    prot = {'Protocolo A (inadequado)': avaliar(pipe(rf()), Xa, ya, cv_a),
            'Protocolo B (adequado)': avaliar(pipe_smote(rf()), X, y, cv, g)}
    tab = pd.DataFrame(prot).T
    print(tab[['recall', 'f1', 'roc_auc']].round(4).to_string())
    R['dif_recall_smote'] = tab.recall.iloc[0] - tab.recall.iloc[1]
    salvar(tab, saida, 'leakage_smote.csv')

    if completo:
        n0 = len(X)
        tr_a, te_a = next(cv_a.split(Xa, ya))
        tr_b, te_b = next(cv.split(X, y, g))
        atl_a = len(set(g.iloc[tr_a[tr_a < n0]]) & set(g.iloc[te_a[te_a < n0]]))
        atl_b = len(set(g.iloc[tr_b]) & set(g.iloc[te_b]))
        onehot = [c for c in X.columns if c not in NUMERICAS]
        sint = Xa.iloc[n0:]
        frac = [((sint[c] > 0.001) & (sint[c] < 0.999)).mean() for c in onehot]
        car = pd.DataFrame({
            'Protocolo A (inadequado)': [len(Xa), len(Xa) - n0, ya.mean(), len(tr_a), len(te_a),
                                         ya.iloc[te_a].mean(), int((te_a >= n0).sum()), atl_a],
            'Protocolo B (adequado)': [n0, 0, y.mean(), len(tr_b), len(te_b),
                                       y.iloc[te_b].mean(), 0, atl_b]},
            index=['registros', 'sinteticos', 'positivos_conjunto', 'treino_por_particao',
                   'teste_por_particao', 'positivos_teste', 'sinteticos_no_teste',
                   'atletas_em_treino_e_teste'])
        print(car.to_string())
        print(f'  sinteticos com valor fracionario nas colunas one-hot: '
              f'{min(frac):.1%} a {max(frac):.1%}')
        salvar(car, saida, 'protocolos_smote.csv')

        # ---------------------------------------------------------------------
        titulo(f'[{rotulo}] 5.4.2  Agrupamento por atleta (Tabela 5.13)')
        cv_n = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        modelo = pipe(rf(class_weight='balanced'))
        grp = {'StratifiedKFold (atleta em treino e teste)': avaliar(modelo, X, y, cv_n),
               'StratifiedGroupKFold (agrupado)': avaliar(modelo, X, y, cv, g)}
        tab = pd.DataFrame(grp).T
        print(tab[['f1', 'roc_auc']].round(4).to_string())
        salvar(tab, saida, 'leakage_grupo.csv')

    salvar(pd.Series(R).to_frame('valor'), saida, 'resumo.csv')
    return X, y, g


# =============================================================================
# EXECUCAO
# =============================================================================
principal = pd.read_csv(ARQ_PRINCIPAL)
soccer = pd.read_csv(ARQ_SOCCER)

titulo('4.1.1 e 5.5  Procedencia: o arquivo de futebol e um recorte da base principal?')
cols = [c for c in principal.columns if c != GROUP]
chave = set(map(tuple, principal[cols].round(6).values))
iguais = sum(1 for r in map(tuple, soccer[cols].round(6).values) if r in chave)
print(f'  linhas do arquivo de futebol presentes na base principal: {iguais} de {len(soccer)}')
print(f'  registros com Sport == "Soccer" na base principal: '
      f'{(principal.Sport == "Soccer").sum()}')
print(f'  atletas em comum: {len(set(principal[GROUP]) & set(soccer[GROUP]))} de '
      f'{soccer[GROUP].nunique()}')

X, y, g = experimento(principal, 'BASE COMPLETA', BASE / 'outputs')

# -----------------------------------------------------------------------------
titulo('5.2.5  Ajuste de limiar com particoes de treino, calibracao e avaliacao (Tabela 5.9)')
i_tr, i_resto = next(GroupShuffleSplit(1, test_size=0.4, random_state=RANDOM_STATE)
                     .split(X, y, g))
j_cal, j_te = next(GroupShuffleSplit(1, test_size=0.5, random_state=RANDOM_STATE)
                   .split(X.iloc[i_resto], y.iloc[i_resto], g.iloc[i_resto]))
i_cal, i_te = i_resto[j_cal], i_resto[j_te]
for nome, idx in [('treino', i_tr), ('calibracao', i_cal), ('avaliacao', i_te)]:
    print(f'  {nome:<11} {len(idx):>6,} registros | {g.iloc[idx].nunique()} atletas')
print(f'  atletas em mais de uma particao: '
      f'{len(set(g.iloc[i_tr]) & set(g.iloc[i_cal])) + len(set(g.iloc[i_tr]) & set(g.iloc[i_te])) + len(set(g.iloc[i_cal]) & set(g.iloc[i_te]))}')
mod = pipe(rf(class_weight='balanced')).fit(X.iloc[i_tr], y.iloc[i_tr])
p_cal = mod.predict_proba(X.iloc[i_cal])[:, 1]
p_te = mod.predict_proba(X.iloc[i_te])[:, 1]
prec, rec, thr = precision_recall_curve(y.iloc[i_cal], p_cal)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
limiar = thr[int(np.argmax(f1s))]
print(f'  limiar escolhido na calibracao: {limiar:.4f}')
linhas = []
for nome, t in [('padrao', 0.5), ('calibrado', limiar)]:
    p = (p_te >= t).astype(int)
    linhas.append({'limiar': nome, 'valor': t,
                   'precisao': precision_score(y.iloc[i_te], p, zero_division=0),
                   'recall': recall_score(y.iloc[i_te], p), 'f1': f1_score(y.iloc[i_te], p)})
lim = pd.DataFrame(linhas)
print(lim.round(4).to_string(index=False))
print(f'  probabilidade < 0,1: {(p_te < 0.1).sum()} | > 0,5: {(p_te > 0.5).sum()} | '
      f'entre o limiar e 0,5: {((p_te >= limiar) & (p_te < 0.5)).sum()}')
salvar(lim, BASE / 'outputs', 'ajuste_limiar.csv', index=False)

# -----------------------------------------------------------------------------
titulo('5.3  Explicabilidade: Gini, permutacao e SHAP (Tabela 5.10)')
import shap          # carregado so aqui; junto do XGBoost em paralelo, causa falha de memoria
FEAT = list(X.columns)
itr, ite = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
                .split(X, y, g))
modelo = pipe(rf(class_weight='balanced')).fit(X.iloc[itr], y.iloc[itr])
gini = pd.Series(modelo.named_steps['clf'].feature_importances_, index=FEAT)
perm = permutation_importance(modelo, X.iloc[ite], y.iloc[ite], scoring='roc_auc',
                              n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
perm = pd.Series(perm.importances_mean, index=FEAT)
idx = np.random.default_rng(RANDOM_STATE).choice(len(ite), size=800, replace=False)
amostra = X.iloc[ite].iloc[idx]
Xs = pd.DataFrame(modelo.named_steps['scaler'].transform(amostra), columns=FEAT)
explicador = shap.TreeExplainer(modelo.named_steps['clf'])
sv = explicador.shap_values(Xs)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
shap_abs = pd.Series(np.abs(sv).mean(0), index=FEAT)
ranks = pd.DataFrame({'gini': gini.rank(ascending=False), 'permutacao': perm.rank(ascending=False),
                      'shap': shap_abs.rank(ascending=False)}).astype(int).sort_values('shap')
print(ranks.to_string())
salvar(pd.DataFrame({'gini': gini, 'permutacao': perm, 'shap_abs_medio': shap_abs})
       .join(ranks.add_prefix('posicao_')), BASE / 'outputs', 'ranking_explicabilidade.csv')

na_regra_am = ((amostra.Heart_Rate > 180) | (amostra.Muscle_Activity > 80)).values
ev = explicador.expected_value
ev = ev[1] if np.ndim(ev) else ev
print(f'  valor esperado do modelo: {ev:.4f}')
print(f'  registros da regra na subamostra de 800: {na_regra_am.sum()} '
      f'(na particao de avaliacao inteira: '
      f'{int(((X.iloc[ite].Heart_Rate > 180) | (X.iloc[ite].Muscle_Activity > 80)).sum())})')
ihr = FEAT.index('Heart_Rate')
print(f'  SHAP de Heart_Rate no registro da regra: {np.round(sv[na_regra_am, ihr], 2)}')
for c in ['Heart_Rate', 'Muscle_Activity']:
    j = FEAT.index(c)
    alto = (amostra[c] >= amostra[c].quantile(0.9)).values & ~na_regra_am
    print(f'  {c}: SHAP medio no decil superior fora da regra {sv[alto, j].mean():+.3f}')
np.save(BASE / 'outputs' / 'shap_valores.npy', sv)
Xs.to_csv(BASE / 'outputs' / 'shap_amostra_padronizada.csv', index=False)

# -----------------------------------------------------------------------------
titulo('5.3.2  Arvore ilustrativa (profundidade 4, unidades originais, base inteira)')
arv = DecisionTreeClassifier(max_depth=4, class_weight='balanced',
                             random_state=RANDOM_STATE).fit(X, y)
folha = arv.apply(X)
folhas = pd.DataFrame({'registros': pd.Series(folha).value_counts(),
                       'taxa_de_risco': pd.Series(y.values).groupby(folha).mean()})
print(folhas.sort_values('taxa_de_risco').round(3).to_string())
salvar(folhas, BASE / 'outputs', 'arvore_folhas.csv', index_label='no')

# -----------------------------------------------------------------------------
Xf, yf, gf = experimento(soccer, 'SUBGRUPO DE FUTEBOL', BASE / 'outputs_soccer', completo=False)

# -----------------------------------------------------------------------------
titulo('[SUBGRUPO DE FUTEBOL] 5.5  Explicabilidade (Tabela 5.14, ultimo achado do texto)')
# Mesmo procedimento da Secao 5.3, aplicado ao subgrupo de futebol.
FEATf = list(Xf.columns)
itr, ite = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
                .split(Xf, yf, gf))
modelo = pipe(rf(class_weight='balanced')).fit(Xf.iloc[itr], yf.iloc[itr])
gini = pd.Series(modelo.named_steps['clf'].feature_importances_, index=FEATf)
perm = permutation_importance(modelo, Xf.iloc[ite], yf.iloc[ite], scoring='roc_auc',
                              n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
perm = pd.Series(perm.importances_mean, index=FEATf)
idx = np.random.default_rng(RANDOM_STATE).choice(len(ite), size=min(800, len(ite)),
                                                 replace=False)
Xs = pd.DataFrame(modelo.named_steps['scaler'].transform(Xf.iloc[ite].iloc[idx]),
                  columns=FEATf)
sv = shap.TreeExplainer(modelo.named_steps['clf']).shap_values(Xs)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
shap_abs = pd.Series(np.abs(sv).mean(0), index=FEATf)
ranks = pd.DataFrame({'gini': gini.rank(ascending=False), 'permutacao': perm.rank(ascending=False),
                      'shap': shap_abs.rank(ascending=False)}).astype(int).sort_values('shap')
print(ranks.head(5).to_string())
salvar(pd.DataFrame({'gini': gini, 'permutacao': perm, 'shap_abs_medio': shap_abs})
       .join(ranks.add_prefix('posicao_')), BASE / 'outputs_soccer', 'ranking_explicabilidade.csv')

titulo('Concluido')
