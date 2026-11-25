from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

BASE=Path(__file__).resolve().parent
OUT=BASE/'generated/figure_3';OUT.mkdir(parents=True,exist_ok=True)
hashes={p:hashlib.sha256(p.read_bytes()).digest() for p in BASE.rglob('*.docx')}
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
blue='#244E73';orange='#AC5D2B';gray='#777E85'
lam=np.array([12.,5.,3.]);corrected=np.array([26/3,2.5,1.]);ratios=lam/corrected
D=np.diag(np.sqrt(ratios[:2]))
rng=np.random.default_rng(20261123)
X=rng.normal(size=(2,90));X-=X.mean(axis=1,keepdims=True)
ev,U=np.linalg.eigh(X@X.T/X.shape[1]);Z=((U/np.sqrt(ev))@U.T)@X
ZD=D@Z;ZR=np.linalg.solve(D,ZD)
assert np.allclose(Z@Z.T/90,np.eye(2))
assert np.allclose(ZD@ZD.T/90,np.diag(ratios[:2]))
assert np.allclose(ZR,Z)
fig=plt.figure(figsize=(9,7.4))
gs=fig.add_gridspec(2,3,left=.09,right=.96,top=.91,bottom=.22,hspace=.61,height_ratios=[1,1.04],wspace=.35)
titles=['(a) Ordinary whitening','(b) Scaling with NR eigenvalues','(c) Rewhitening']
formulas=[r'$\widehat{C}_Z=I$',r'$\widehat{C}_{Z_{\rm NR}}=D^2\ne I$',r'$\widehat{C}_{Z_{\rm R}}=I$']
for k,(data,scale) in enumerate(zip([Z,ZD,ZR],[np.ones(2),np.sqrt(ratios[:2]),np.ones(2)])):
    ax=fig.add_subplot(gs[0,k]);color=orange if k==1 else blue
    ax.set_aspect('equal');ax.set_xlim(-2.4,2.4);ax.set_ylim(-2.4,2.4)
    # Mesma nuvem, três transformações; elipses representam a covariância.
    ax.scatter(data[0],data[1],s=8,c=color,alpha=.28,linewidths=0)
    ax.add_patch(Ellipse((0,0),2,2,fill=False,lw=.8,ls=(0,(3,3)),edgecolor=gray,zorder=2))
    ax.add_patch(Ellipse((0,0),2*scale[0],2*scale[1],fill=False,lw=1.7,edgecolor=color,zorder=3))
    for spine in ['left','bottom']:
        ax.spines[spine].set_position('zero');ax.spines[spine].set_color('#A7ADB2');ax.spines[spine].set_linewidth(.65)
    ax.set_xticks([-1,1]);ax.set_yticks([-1,1]);ax.tick_params(length=2,labelsize=9,pad=3)
    ax.set_title(titles[k],fontsize=11,pad=17)
    ax.text(.5,-.15,formulas[k],transform=ax.transAxes,ha='center',va='top',fontsize=13)

ax=fig.add_subplot(gs[1,:]);x=np.arange(3)
colors=[blue,orange,'#637C55'];markers=['o','^','s']
for j in range(3):
    yy=[1,ratios[j],1]
    ax.plot(x,yy,color=colors[j],marker=markers[j],mfc='white',mew=1.4,ms=6,lw=1.2,label=f'PC{j+1}')
    ax.annotate(['18/13 ≈ 1.385','2','3'][j],(1,ratios[j]),xytext=(9,2),textcoords='offset points',color=colors[j],fontsize=10)
ax.axhline(1,color=gray,lw=.8,ls=(0,(3,3)),zorder=0)
ax.set_xlim(-.16,2.16);ax.set_ylim(.74,3.28)
ax.set_xticks(x,['Ordinary','NR scaling','Rewhitened']);ax.set_yticks([1,2,3])
ax.set_ylabel('Sample variance',labelpad=8)
ax.set_title('(d) Variances of the three components',loc='left',fontsize=12,pad=14)
ax.legend(loc='upper right',ncol=3,frameon=False,handlelength=1.6,columnspacing=1.3)
ax.tick_params(length=3,color=gray);ax.spines['left'].set_color(gray);ax.spines['bottom'].set_color(gray)
fig.text(.525,.105,r'$Z_{\rm NR}=DZ\qquad\longrightarrow\qquad Z_{\rm R}=(D^2)^{-1/2}DZ=Z$',ha='center',fontsize=16)
fig.text(.525,.052,r'$D_{jj}=\sqrt{\widehat{\lambda}_j/\widetilde{\lambda}_j}>0$'+ '  ·  same basis and retained components',ha='center',fontsize=11,color='#454B50')
stem=OUT/'figura3_cancelamento_geometrico'
for suffix in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+suffix),dpi=450,facecolor='white')
fig.savefig(OUT/'previa.png',dpi=150,facecolor='white');plt.close(fig)
caption='Figura 3. Cancelamento de uma reescala positiva pelo rebranqueamento. (a–c) A mesma nuvem bidimensional é mostrada após branqueamento usual, reescala com autovalores corrigidos e rebranqueamento. Os contornos contínuos representam elipses de covariância com semieixos iguais aos desvios-padrão; o círculo tracejado é a referência unitária. Os pontos são ilustrativos, e os contornos não delimitam todos os pontos. (d) Exemplo algébrico com autovalores observados (12, 5, 3, 2) e autovalores NR (26/3, 5/2, 1) nas três componentes retidas. A substituição na escala inversa produz variâncias 18/13, 2 e 3; o rebranqueamento na mesma base as devolve a 1 e cancela exatamente a reescala. O resultado pressupõe fatores positivos, mesma amostra, mesma normalização e mesmo subespaço retido. Não representa uma medida de redução de ruído ou de desempenho de separação.'
(OUT/'legenda_figura3.txt').write_text(caption,encoding='utf-8')
for p,h in hashes.items():assert hashlib.sha256(p.read_bytes()).digest()==h
print(f'Figura gerada. {len(hashes)} arquivos Word verificados sem alteração. Cancelamento numérico verificado.')
