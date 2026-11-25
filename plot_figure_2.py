from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
BASE=Path(__file__).resolve().parent
OUT=BASE/'generated/figure_2';OUT.mkdir(parents=True,exist_ok=True)
hashes={p:hashlib.sha256(p.read_bytes()).digest() for p in BASE.rglob('*.docx')}
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
blue='#244E73';gray='#747E87';L=7
fig,axs=plt.subplots(1,4,figsize=(11,5.4));fig.subplots_adjust(left=.045,right=.985,top=.75,bottom=.40,wspace=.30)
for ax,lag,title in zip(axs,[0,1,3,7],['(a) Lag $\\ell=0$','(b) Lag $\\ell=1$','(c) Lag $\\ell=3$','(d) Lag $\\ell\\geq L$']):
    C=np.eye(L,k=-lag,dtype=int)
    assert C.sum()==max(L-lag,0)
    ax.imshow(C,cmap=ListedColormap(['#FFFFFF',blue]),vmin=0,vmax=1,interpolation='nearest')
    for i in range(L):
        for j in range(L):ax.text(j,i,str(C[i,j]),ha='center',va='center',color='white' if C[i,j] else '#B1B8BE',fontsize=9)
    ax.set_xticks(range(L),range(1,L+1));ax.set_yticks(range(L),range(1,L+1));ax.tick_params(length=0,pad=5,labelsize=9)
    ax.set_xticks(np.arange(-.5,L,1),minor=True)
    ax.set_yticks(np.arange(-.5,L,1),minor=True);ax.grid(which='minor',color='#DFE4E8',lw=.5);ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values():spine.set_color('#AAB3BB');spine.set_linewidth(.7)
    ax.set_title(title,fontsize=12,pad=15);ax.set_xlabel('Coordinate $j$',labelpad=7)
    ax.set_ylabel('Coordinate $i$',labelpad=6)
    expr=[r'$R(0)=I_L$',r'$R(1)=J_1$',r'$R(3)=J_3$',r'$R(\ell)=0$'][[0,1,3,7].index(lag)]
    ax.text(.5,-.36,expr,transform=ax.transAxes,ha='center',fontsize=14,color=blue)
fig.text(.515,.95,'Noise covariance after delay embedding',ha='center',va='top',fontsize=17)
fig.text(.515,.855,'$e_t\\ \\mathrm{white},\\quad\\mathrm{Var}(e_t)=1,\\qquad n_t=(e_t,e_{t-1},\\ldots,e_{t-L+1})^{\\mathsf{T}}$',ha='center',fontsize=12,color='#555E66')
fig.text(.515,.13,r'$R(\ell)=\mathbb{E}[n_tn_{t-\ell}^{\mathsf{T}}],\qquad R_{ij}(\ell)=\mathbf{1}\{i-j=\ell\}$',ha='center',fontsize=15)
fig.text(.515,.045,'Example with $L=7$ · the diagonal shifts and vanishes for $|\\ell|\\geq L$'+ '\n'+'For negative lags: $R(-\\ell)=R(\\ell)^{\\mathsf{T}}$',ha='center',fontsize=11,color='#555E66',linespacing=1.4)
stem=OUT/'figura_matrizes_ruido'
for suffix in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+suffix),dpi=450,facecolor='white')
fig.savefig(OUT/'previa.png',dpi=150,facecolor='white');plt.close(fig)
(OUT/'legenda.txt').write_text('Estrutura da covariância do ruído branco escalar após imersão por atrasos. Para n_t = (e_t, e_{t−1}, …, e_{t−L+1})ᵀ, define-se R(ℓ) = E[n_t n_{t−ℓ}ᵀ]. Com variância unitária, a entrada (i,j) é 1 quando i−j = ℓ e zero nos demais casos. Mostram-se matrizes populacionais exatas para L = 7: (a) identidade no atraso zero; (b,c) diagonais deslocadas nos atrasos 1 e 3; (d) matriz nula para ℓ ≥ L. Para atrasos negativos, a matriz é a transposta da correspondente matriz de atraso positivo. A convenção de covariância defasada determina o sentido do deslocamento. Este painel é esquemático e usa L = 7 para legibilidade; não modifica os experimentos do artigo com L = 12.',encoding='utf-8')
for p,h in hashes.items():assert hashlib.sha256(p.read_bytes()).digest()==h
print(f'Matrizes verificadas; {len(hashes)} arquivos Word preservados.')

