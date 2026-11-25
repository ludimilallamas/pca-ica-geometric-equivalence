from pathlib import Path
import csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator,FixedLocator,FixedFormatter
BASE=Path(__file__).resolve().parent
OUT=BASE/'generated/figure_5';OUT.mkdir(parents=True,exist_ok=True)
hashes={p:hashlib.sha256(p.read_bytes()).digest() for p in BASE.rglob('*.docx')}
with (BASE/'reference_noise_covariance.csv').open(encoding='utf-8-sig',newline='') as f:rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':12,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors=['#244E73','#397A68','#AC5D2B','#84617D'];markers=['o','s','^','D'];lags=[0,1,3,12]
fig,axs=plt.subplots(1,2,figsize=(10,5.5));fig.subplots_adjust(left=.09,right=.96,top=.85,bottom=.32,wspace=.37)
handles=[]
for lag,color,marker in zip(lags,colors,markers):
    sub=sorted([r for r in rows if r['lag']==lag],key=lambda r:r['T']);ts=[r['T'] for r in sub]
    assert len(sub)==3 and all(r['repeticoes']==200 for r in sub)
    target=np.sqrt(max(12-lag,0)/12)
    assert all(np.isclose(r['norma_alvo'],target) for r in sub)
    h=axs[0].errorbar(ts,[r['erro_medio'] for r in sub],yerr=[1.96*r['erro_padrao_monte_carlo'] for r in sub],marker=marker,ms=5,mfc='white',mew=1.1,lw=1.2,capsize=3,color=color,label=rf'$\ell={lag}$',alpha=.9)
    handles.append(h)
    axs[1].plot(ts,[r['norma_media'] for r in sub],marker=marker,ms=5,mfc='white',mew=1.2,lw=1.4,color=color)
    axs[1].axhline(target,color=color,ls=(0,(3,3)),lw=.85,alpha=.65)
    label={0:r'$1$',1:r'$\sqrt{11/12}$',3:r'$\sqrt{9/12}$',12:r'$0$'}[lag]
    offset={0:10,1:-5,3:-8,12:5}[lag]
    axs[1].annotate(label,(4200,target),xytext=(4,offset),textcoords='offset points',color=color,fontsize=11,va='center')
for ax in axs:
    ax.set_xscale('log',base=2);ax.set_xticks([256,1024,4096],['256','1,024','4,096']);ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Record length $T$',labelpad=9);ax.tick_params(length=3,color='#899198');ax.spines['left'].set_color('#899198');ax.spines['bottom'].set_color('#899198')
    ax.grid(axis='y',lw=.6,color='#E6E9EB');ax.set_axisbelow(True)
axs[0].set_xlim(225,4600);axs[0].set_ylim(0,.255);axs[0].set_yticks([0,.05,.1,.15,.2,.25],['0','0.05','0.10','0.15','0.20','0.25'])
axs[0].set_ylabel('Mean $\\|\\widehat{R}(\\ell)-R(\\ell)\\|_F/\\sqrt{L}$',labelpad=8)
axs[0].set_title('(a) Estimation error decreases',loc='left',pad=17,fontsize=13)
axs[0].text(.53,.80,'Nearly overlapping\ncurves',transform=axs[0].transAxes,color='#646D75',fontsize=11,ha='center')
axs[1].set_xlim(225,8500);axs[1].set_ylim(-.045,1.14);axs[1].set_yticks([0,.25,.5,.75,1],['0','0.25','0.50','0.75','1'])
axs[1].set_ylabel('Mean $\\|\\widehat{R}(\\ell)\\|_F/\\sqrt{L}$',labelpad=8)
axs[1].set_title('(b) Covariance may persist',loc='left',pad=17,fontsize=13)
fig.legend(handles=handles,loc='center',bbox_to_anchor=(.52,.19),ncol=4,frameon=False,fontsize=12,handlelength=2,columnspacing=2)
fig.text(.52,.105,r'$\|R(\ell)\|_F/\sqrt{L}=\sqrt{\max(L-|\ell|,0)/L}$',ha='center',fontsize=15)
fig.text(.52,.035,'Scalar white noise with unit variance · $L=12$ · 200 records per length'+ '\n'+'Bars: $\\pm1.96$ Monte Carlo standard errors · dashed lines: population norms',ha='center',fontsize=10.5,color='#555E66',linespacing=1.4)
stem=OUT/'figura_covariancia_defasada'
for suffix in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+suffix),dpi=450,facecolor='white')
fig.savefig(OUT/'previa.png',dpi=150,facecolor='white');plt.close(fig)
(OUT/'legenda.txt').write_text('Erro de estimação e persistência da covariância do ruído após imersão por atrasos. (a) Média da distância normalizada à matriz populacional correta R(ℓ). As quatro curvas quase coincidem e diminuem com o comprimento do registro. As barras preservam ±1,96 Monte Carlo standard errors do experimento original. (b) Média da norma normalizada da matriz estimada. Os tracejados indicam as normas populacionais: 1, √(11/12), √(9/12) e 0, respectivamente, para ℓ = 0, 1, 3 e 12. Para ruído branco escalar com variância unitária e L = 12, a imersão gera matrizes de deslocamento não nulas para |ℓ| < L. Assim, convergir para a matriz populacional correta não implica convergir para a matriz zero. Foram preservados os resultados originais de 200 records per length em resultados_monte_carlo.csv; não se realizaram novas simulações.',encoding='utf-8')
for p,h in hashes.items():assert hashlib.sha256(p.read_bytes()).digest()==h
print(f'12 resultados originais preservados; {len(hashes)} arquivos Word sem alterações; normas populacionais verificadas.')
