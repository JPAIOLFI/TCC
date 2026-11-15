# -*- coding: utf-8 -*-
"""
Gera as figuras da tese a partir dos resultados de pipeline.py.
Rode pipeline.py antes. As figuras sao gravadas em fig_leve/, com os nomes
usados no LaTeX. Todas sao desenhadas na largura final da pagina (16 cm) e
usam virgula decimal.

Uso:  python figuras.py
"""
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (precision_recall_curve, average_precision_score, confusion_matrix,
                             precision_score, recall_score, f1_score)
import shap

BASE = Path(__file__).resolve().parent
OUT, OUT_S, DEST = BASE / 'outputs', BASE / 'outputs_soccer', BASE / 'fig_leve'
DEST.mkdir(exist_ok=True)
RS, LARG = 42, 6.3
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8.5,
    'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})
AZUL, VERM, CINZA, VERDE = '#2E6F9E', '#C0503A', '#9A9A9A', '#2E7D32'


def br(v, c=2):
    s = f'{v:,.{c}f}'
    return s.replace(',', 'X').replace('.', ',').replace('X', '.')


def eixo_br(ax, eixo='x', c=2):
    f = FuncFormatter(lambda v, _: br(v, c))
    (ax.xaxis if eixo == 'x' else ax.yaxis).set_major_formatter(f)


def salvar(fig, nome):
    fig.savefig(DEST / nome)
    plt.close(fig)
    print('  ', nome)


def ler(pasta, nome, **kw):
    return pd.read_csv(pasta / nome, encoding='utf-8-sig', **kw)


df = pd.read_csv(BASE / 'data' / 'athlete_physiological_dataset.csv')
y = df['Injury_Risk'].astype(int)
g = df['Athlete_ID']
X = df.drop(columns=['Injury_Risk', 'Athlete_ID'])
X = pd.get_dummies(X, columns=X.select_dtypes('object').columns.tolist(), drop_first=True, dtype=float)
REF = ler(OUT, 'resumo.csv', index_col=0).valor
REF_S = ler(OUT_S, 'resumo.csv', index_col=0).valor

# ---------------------------------------------------------------- 5.1  alvo
n1, n0 = int(y.sum()), int((y == 0).sum())
fig, ax = plt.subplots(figsize=(4.1, 2.6))
b = ax.bar(['Sem risco (0)', 'Risco (1)'], [n0, n1], color=[AZUL, VERM], width=0.55)
for barra, n in zip(b, [n0, n1]):
    ax.text(barra.get_x() + barra.get_width() / 2, n + 350,
            f'{br(n, 0)}\n({br(100 * n / len(y), 1)}%)', ha='center', va='bottom')
ax.set_ylabel('Número de registros'); ax.set_ylim(0, n0 * 1.18); eixo_br(ax, 'y', 0)
salvar(fig, '26_distribuicao_alvo.png')

# ---------------------------------------------------------------- 5.2  d de Cohen
ef = ler(OUT, 'tamanho_efeito_features.csv', index_col=0)
ordem = ['Heart_Rate', 'Muscle_Activity', 'Oxygen_Saturation', 'Age', 'Motion_Z', 'Motion_X', 'Motion_Y']
s = ef.d_cohen.reindex(ordem[::-1])
fig, ax = plt.subplots(figsize=(LARG * 0.75, 2.7))
ax.barh(s.index, s.values, color=[VERM if abs(v) >= 0.2 else AZUL for v in s.values], height=0.6)
for lim, est, rot in [(0.2, ':', 'efeito pequeno (|d| = 0,2)'), (0.5, '--', 'efeito médio (|d| = 0,5)')]:
    ax.axvline(lim, color=CINZA, ls=est, lw=0.9, label=rot)
    ax.axvline(-lim, color=CINZA, ls=est, lw=0.9)
ax.axvline(0, color='black', lw=0.6); ax.set_xlim(-0.55, 0.55)
ax.set_xlabel('$d$ de Cohen (classe 1 menos classe 0)'); eixo_br(ax, 'x', 1)
ax.legend(loc='upper left', frameon=False)
salvar(fig, '27_cohen_d.png')

# ---------------------------------------------------------------- 5.3  regra
fig, ax = plt.subplots(figsize=(LARG * 0.85, 3.6))
for classe, cor, tam, alfa, rot in [(0, AZUL, 3, 0.25, 'Sem risco'), (1, VERM, 5, 0.7, 'Risco')]:
    d = df[y == classe]
    ax.scatter(d.Heart_Rate, d.Muscle_Activity, s=tam, alpha=alfa, color=cor, lw=0, label=rot)
ax.axvline(180, color='black', ls='--', lw=1, label='Heart_Rate = 180')
ax.axhline(80, color='black', ls='-.', lw=1, label='Muscle_Activity = 80')
ax.set_xlabel('Heart_Rate (bpm)'); ax.set_ylabel('Muscle_Activity (%)')
eixo_br(ax, 'x', 0); eixo_br(ax, 'y', 0)
ax.legend(loc='lower left', frameon=True, framealpha=1, fontsize=6.8, markerscale=2.5)
salvar(fig, '41_regra_deterministica.png')

# ---------------------------------------------------------------- 5.4 e 5.5  11 configuracoes
res = ler(OUT, 'resultados_comparativos.csv', index_col=0)
ordem = ['[Baseline] Random Forest', '[Weights] RF', '[SMOTE] RF', '[Weights] DT',
         '[Baseline] Árvore de Decisão', '[SMOTE] DT', '[SMOTE] LR', '[Weights] LR',
         '[Baseline] Dummy (Estratificado)', '[Baseline] Regressão Logística',
         '[Baseline] Dummy (Majoritário)']
t = res.loc[ordem][::-1]
rotulos = [i.replace('(Estratificado)', '(Estrat.)') for i in t.index]
ref = REF.f1_regra_particoes
cores = [CINZA if 'Dummy' in c else AZUL for c in t.index]
fig, axs = plt.subplots(1, 3, figsize=(LARG, 3.3), sharey=True)
for ax, (col, dp, tit, xmax) in zip(axs, [('recall', 'recall_std', 'Recall', 0.6),
                                          ('f1', 'f1_std', 'F1-Score', 0.25),
                                          ('roc_auc', None, 'ROC-AUC', 0.6)]):
    ax.barh(rotulos, t[col], xerr=None if dp is None else t[dp], color=cores, height=0.65,
            error_kw=dict(lw=0.7, capsize=1.5, ecolor='#333333'))
    ax.set_title(tit); ax.set_xlim(0, xmax); eixo_br(ax, 'x', 2 if xmax < 0.3 else 1)
    ax.grid(axis='x', lw=0.4, alpha=0.5)
l1 = axs[1].axvline(ref, color=VERDE, ls='--', lw=1, label=f'regra nas mesmas partições (F1 = {br(ref, 4)})')
l2 = axs[2].axvline(0.5, color='black', ls=':', lw=0.9, label='ordenação aleatória (ROC-AUC = 0,5)')
fig.legend(handles=[l1, l2, Patch(color=CINZA, label='classificadores de referência (Dummy)')],
           loc='lower center', ncol=3, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, -0.04))
axs[0].tick_params(axis='y', labelsize=7)
fig.tight_layout(w_pad=0.6, rect=(0, 0.05, 1, 1))
salvar(fig, '34_comparativo_metricas.png')

fig, ax = plt.subplots(figsize=(LARG * 0.8, 3.2))
ax.barh(rotulos, t.f1, color=[VERM if 'Dummy' in c else AZUL for c in t.index], height=0.65)
ax.axvline(ref, color=VERDE, ls='--', lw=1.1, label=f'regra nas mesmas partições (F1 = {br(ref, 4)})')
ax.set_xlabel('F1-Score (média de 10 partições)'); ax.set_xlim(0, 0.21)
ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20]); eixo_br(ax, 'x', 2)
ax.grid(axis='x', lw=0.4, alpha=0.5); ax.legend(loc='lower right', frameon=True, fontsize=7)
ax.tick_params(axis='y', labelsize=7)
salvar(fig, '35_referencia_desempenho.png')

# ---------------------------------------------------------------- 5.6  robustez
def painel(ax, pasta, ref, titulo):
    d = ler(pasta, 'boosting_e_tuning.csv', index_col=0)
    nomes = ['[Tuned] Random Forest (aninhada)', '[Boosting] XGBoost',
             '[Boosting] HistGradientBoosting', '[Boosting] XGBoost + scale_pos_weight']
    d = d.loc[nomes]
    rot = ['Random Forest\notimizado\n(aninhada)', 'XGBoost', 'HistGradient-\nBoosting',
           'XGBoost +\nscale_pos_\nweight']
    x = np.arange(4)
    ax.bar(x, d.f1, yerr=d.f1_std, color=[AZUL] + [VERM] * 3, width=0.6,
           error_kw=dict(lw=0.8, capsize=2.5, ecolor='#333333'))
    ax.axhline(ref, color=VERDE, ls='--', lw=1.1)
    for xi, (v, s) in enumerate(zip(d.f1, d.f1_std)):
        ax.text(xi, max(v + s, ref + 0.006) + 0.012, f'{br(v, 4)}\n({br(100 * v / ref, 2)}%)',
                ha='center', va='bottom', fontsize=6.8)
    ax.set_xticks(x, rot, fontsize=6.8)
    ax.set_title(f'{titulo}\nregra nas mesmas partições = {br(ref, 4)}')
    ax.set_ylim(0, 0.46); eixo_br(ax, 'y', 2); ax.grid(axis='y', lw=0.4, alpha=0.5)


fig, axs = plt.subplots(1, 2, figsize=(LARG, 3.1))
painel(axs[0], OUT, REF.f1_regra_particoes, 'Base completa (23.400 registros)')
painel(axs[1], OUT_S, REF_S.f1_regra_particoes, 'Subgrupo de futebol (5.100 registros)')
axs[0].set_ylabel('F1-Score (média de 10 partições)')
fig.tight_layout(w_pad=1.5)
salvar(fig, '36_robustez_referencia.png')

# ---------------------------------------------------------------- 5.7 e 5.8  limiar
i_tr, i_resto = next(GroupShuffleSplit(1, test_size=0.4, random_state=RS).split(X, y, g))
j_cal, j_te = next(GroupShuffleSplit(1, test_size=0.5, random_state=RS)
                   .split(X.iloc[i_resto], y.iloc[i_resto], g.iloc[i_resto]))
i_cal, i_te = i_resto[j_cal], i_resto[j_te]
mod = Pipeline([('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                               random_state=RS, n_jobs=-1))]).fit(X.iloc[i_tr], y.iloc[i_tr])
p_cal = mod.predict_proba(X.iloc[i_cal])[:, 1]
p_te = mod.predict_proba(X.iloc[i_te])[:, 1]
yc, yt = y.iloc[i_cal].values, y.iloc[i_te].values
prec, rec, thr = precision_recall_curve(yc, p_cal)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
k = int(np.argmax(f1s)); melhor = thr[k]
fig, axs = plt.subplots(1, 2, figsize=(LARG, 2.6))
ax = axs[0]
ax.plot(thr, prec[:-1], label='Precisão', lw=1.3); ax.plot(thr, rec[:-1], label='Recall', lw=1.3)
ax.plot(thr, f1s, label='F1-Score', lw=1.3, ls='--')
ax.axvline(0.5, color='gray', ls=':', lw=0.9, label='padrão (0,5)')
ax.axvline(melhor, color='red', ls='--', lw=0.9, label=f'selecionado ({br(melhor, 3)})')
ax.set_xlabel('Limiar de decisão'); ax.set_ylabel('Valor da métrica'); ax.set_title('Métricas por limiar (calibração)')
ax.legend(fontsize=6, frameon=True, framealpha=1, loc='center right', bbox_to_anchor=(1.0, 0.55))
eixo_br(ax, 'x', 1); eixo_br(ax, 'y', 1); ax.grid(lw=0.4, alpha=0.5)
ax = axs[1]
ax.plot(rec, prec, lw=1.3)
ax.scatter(rec[k], prec[k], color='red', zorder=5, s=14, label=f'limiar selecionado (F1 = {br(f1s[k], 3)})')
ax.axhline(yc.mean(), color='gray', ls=':', lw=0.9, label=f'prevalência ({br(yc.mean(), 3)})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precisão')
ax.set_title(f'Curva precisão-recall (AP = {br(average_precision_score(yc, p_cal), 4)})')
ax.legend(fontsize=6.3, frameon=False, loc='upper right'); eixo_br(ax, 'x', 1); eixo_br(ax, 'y', 1)
ax.grid(lw=0.4, alpha=0.5)
fig.tight_layout(w_pad=1.5)
salvar(fig, '39_curva_limiar_calibracao.png')

fig, axs = plt.subplots(1, 2, figsize=(LARG, 2.7))
for ax, lim, rot, cmap in [(axs[0], 0.5, 'Limiar padrão (0,500)', 'Blues'),
                           (axs[1], melhor, f'Limiar calibrado ({br(melhor, 3)})', 'Oranges')]:
    pred = (p_te >= lim).astype(int)
    cm = confusion_matrix(yt, pred)
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max() * 1.1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, br(v, 0), ha='center', va='center', fontsize=8,
                color='white' if v > cm.max() / 2 else 'black')
    ax.set_xticks([0, 1], ['Sem risco', 'Risco']); ax.set_yticks([0, 1], ['Sem risco', 'Risco'])
    ax.set_xlabel('Classe prevista'); ax.set_ylabel('Classe verdadeira')
    ax.set_title(f'{rot}\nRecall {br(recall_score(yt, pred), 4)} | Precisão '
                 f'{br(precision_score(yt, pred, zero_division=0), 4)} | F1 {br(f1_score(yt, pred), 4)}',
                 fontsize=7.2)
    for sp in ax.spines.values():
        sp.set_visible(True)
fig.tight_layout(w_pad=2)
salvar(fig, '40_matrizes_confusao_avaliacao.png')

# ---------------------------------------------------------------- 5.9  arvore ilustrativa
arv = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=RS).fit(X, y)
T = arv.tree_
folha_de = arv.apply(X)
taxa = pd.Series(y.values).groupby(folha_de).mean()
qtd = pd.Series(folha_de).value_counts()
pos, prox = {}, [0]


def layout(no, prof):
    if T.children_left[no] == -1:
        pos[no] = (prox[0], -prof); prox[0] += 1
    else:
        layout(T.children_left[no], prof + 1); layout(T.children_right[no], prof + 1)
        pos[no] = ((pos[T.children_left[no]][0] + pos[T.children_right[no]][0]) / 2, -prof)


layout(0, 0)
fig, ax = plt.subplots(figsize=(LARG, 3.6)); ax.axis('off')
cmap = plt.get_cmap('Reds')
for no, (x0, y0) in pos.items():
    esq, dirt = T.children_left[no], T.children_right[no]
    if esq != -1:
        for filho, rot in [(esq, 'sim'), (dirt, 'não')]:
            x1, y1 = pos[filho]
            ax.plot([x0, x1], [y0 - 0.14, y1 + 0.16], color='#777777', lw=0.7, zorder=1)
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.02, rot, fontsize=6, color='#555555',
                    ha='center', va='center', bbox=dict(fc='white', ec='none', pad=0.3))
        ax.text(x0, y0, f'{X.columns[T.feature[no]]}\n≤ {br(T.threshold[no], 1)}', ha='center',
                va='center', fontsize=6.6, zorder=2,
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#444444', lw=0.7))
    else:
        r, n = taxa[no], qtd[no]
        pct = f'{100 * r:.0f}' if r in (0.0, 1.0) else br(100 * r, 1)
        ax.text(x0, y0, f'{br(n, 0)} reg.\nrisco {pct}%', ha='center', va='center', fontsize=6.0,
                zorder=2, color='white' if r > 0.6 else 'black',
                bbox=dict(boxstyle='round,pad=0.35', fc=cmap(0.12 + 0.85 * r), ec='#444444', lw=0.7))
xs = [p[0] for p in pos.values()]
ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
ax.set_ylim(min(p[1] for p in pos.values()) - 0.45, 0.4)
salvar(fig, '30_arvore_decisao_legivel.png')

# ---------------------------------------------------------------- 5.10 e 5.11  SHAP
sv = np.load(OUT / 'shap_valores.npy')
Xs = pd.read_csv(OUT / 'shap_amostra_padronizada.csv')
shap.summary_plot(sv, Xs, show=False, max_display=Xs.shape[1], plot_size=(5.2, 4.6))
fig = plt.gcf(); ax = fig.axes[0]
ax.set_xlabel('Valor SHAP (efeito sobre a probabilidade estimada de risco)', fontsize=8)
eixo_br(ax, 'x', 1); ax.tick_params(labelsize=7.5)
cb = fig.axes[-1]; cb.set_yticklabels(['Baixo', 'Alto'], fontsize=7); cb.set_ylabel('Valor da variável', fontsize=7.5)
salvar(fig, '32_shap_beeswarm.png')
shap.summary_plot(sv, Xs, plot_type='bar', show=False, max_display=Xs.shape[1], plot_size=(5.0, 4.2))
fig = plt.gcf(); ax = fig.axes[0]
ax.set_xlabel('Média do valor SHAP absoluto', fontsize=8); eixo_br(ax, 'x', 2); ax.tick_params(labelsize=7.5)
salvar(fig, '33_shap_bar.png')

# ---------------------------------------------------------------- 5.12  protocolos
sm = ler(OUT, 'leakage_smote.csv', index_col=0)
gr = ler(OUT, 'leakage_grupo.csv', index_col=0)
fig, axs = plt.subplots(1, 2, figsize=(LARG, 2.9), sharey=True)
for ax, d, rot, tit in [
        (axs[0], sm, ['Protocolo A\n(SMOTE antes da divisão,\nsem agrupamento)',
                      'Protocolo B\n(SMOTE só no treino,\nagrupado por atleta)'], 'Protocolos de reamostragem'),
        (axs[1], gr, ['StratifiedKFold\n(atleta em treino e teste)',
                      'StratifiedGroupKFold\n(agrupado por atleta)'], 'Agrupamento por atleta')]:
    x = np.arange(2)
    for kk, (mm, nome, cor) in enumerate([('recall', 'Recall', VERM), ('f1', 'F1-Score', AZUL),
                                          ('roc_auc', 'ROC-AUC', VERDE)]):
        v = d[mm].values
        ax.bar(x + (kk - 1) * 0.25, v, 0.24, color=cor, label=nome)
        for xi, vi in zip(x, v):
            ax.text(xi + (kk - 1) * 0.25, vi + 0.015, br(vi, 2), ha='center', va='bottom', fontsize=6.5)
    ax.set_xticks(x, rot, fontsize=6.6); ax.set_title(tit); ax.set_ylim(0, 1.12)
    eixo_br(ax, 'y', 1); ax.grid(axis='y', lw=0.4, alpha=0.5)
axs[0].set_ylabel('Valor da métrica'); axs[1].legend(loc='upper right', frameon=False)
fig.tight_layout(w_pad=1.2)
salvar(fig, '37_protocolos_validacao.png')

# ---------------------------------------------------------------- 4.1  diagrama
fig, ax = plt.subplots(figsize=(LARG, 2.5)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 4)


def caixa(x, y_, w, h, txt, cor='#EAF1F7', ec=AZUL):
    ax.add_patch(FancyBboxPatch((x, y_), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
                                fc=cor, ec=ec, lw=0.9))
    ax.text(x + w / 2, y_ + h / 2, txt, ha='center', va='center', fontsize=6.9)


def seta(x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', lw=0.9, color='#444444'))


caixa(0.05, 1.45, 1.4, 1.1, 'Base\n23.400 registros\n234 atletas', '#F3F3F3', '#666666')
caixa(1.75, 1.45, 1.6, 1.1, 'One-hot das\ncategóricas\n(base inteira,\nsem estatísticas)', '#F3F3F3', '#666666')
caixa(3.65, 1.45, 1.6, 1.1, 'StratifiedGroup-\nKFold, k = 10,\nagrupado por\nAthlete_ID')
caixa(5.6, 2.3, 2.3, 1.5, 'Treino (cerca de 90%)\n1. StandardScaler: ajuste\n2. SMOTE, se aplicável\n3. ajuste do modelo', '#FBEDEA', VERM)
caixa(5.6, 0.2, 2.3, 1.5, 'Teste (cerca de 10%)\n1. StandardScaler:\napenas transforma\n2. predição, sem SMOTE', '#EAF4EA', VERDE)
caixa(8.3, 1.45, 1.65, 1.1, 'Métricas por\npartição;\nmédia das 10')
seta(1.45, 2.0, 1.75, 2.0); seta(3.35, 2.0, 3.65, 2.0)
seta(5.25, 2.25, 5.6, 2.9); seta(5.25, 1.75, 5.6, 1.1)
seta(6.75, 2.3, 6.75, 1.7)
ax.text(6.85, 2.0, 'modelo\najustado', fontsize=6.2, color='#444444', va='center')
seta(7.9, 0.95, 8.3, 1.75)
salvar(fig, '38_diagrama_pipeline.png')
print('pronto')
