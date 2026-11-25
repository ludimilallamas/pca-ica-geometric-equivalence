from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE=Path(__file__).resolve().parent
OUT=BASE/'generated/figure_4';OUT.mkdir(parents=True,exist_ok=True)
hashes={p:hashlib.sha256(p.read_bytes()).digest() for p in BASE.rglob('*.docx')}
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':12,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
blue='#244E73';orange='#AC5D2B';green='#397A68';gray='#899198'
theta=np.linspace(0,2*np.pi,1600,endpoint=False);x=np.cos(theta);y=np.cos(2*theta)
assert abs(np.mean(x*y))<1e-12 and np.allclose(y,2*x*x-1)
fig,axs=plt.subplots(1,2,figsize=(9.6,5.6));fig.subplots_adjust(left=.09,right=.96,top=.84,bottom=.33,wspace=.36)
ax=axs[0];tt=np.linspace(0,2*np.pi,1000)
ax.plot(tt,np.cos(tt),color=blue,lw=1.6,label=r'$X=\cos\theta$')
ax.plot(tt,np.cos(2*tt),color=orange,lw=1.5,label=r'$Y=\cos2\theta$')
ax.axhline(0,color=gray,lw=.7)
ax.set_xlim(0,2*np.pi);ax.set_ylim(-1.18,1.18)
ax.set_xticks([0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],['0',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$'])
ax.set_yticks([-1,0,1]);ax.set_xlabel('Phase $\\theta$',labelpad=9);ax.set_ylabel('Amplitude',labelpad=8)
ax.set_title('(a) Orthogonal harmonics',loc='left',pad=17,fontsize=13)
ax.legend(loc='lower center',bbox_to_anchor=(.5,-.34),ncol=2,frameon=False,fontsize=12,handlelength=1.8,columnspacing=1.1)
ax=axs[1];xx=np.linspace(-1,1,600)
ax.plot(xx,2*xx*xx-1,color=green,lw=1.8)
# Mesma fase fornece um ponto em cada curva temporal e um par na parábola.
phase=np.pi/3;xx0=np.cos(phase);yy0=np.cos(2*phase)
axs[0].axvline(phase,color=gray,ls=(0,(3,3)),lw=.8)
axs[0].scatter([phase],[xx0],s=36,facecolors='white',edgecolors=blue,lw=1.4,zorder=5)
axs[0].scatter([phase],[yy0],s=36,marker='s',facecolors='white',edgecolors=orange,lw=1.4,zorder=5)
ax.plot([xx0,xx0],[0,yy0],color=gray,lw=.8,ls=':');ax.plot([0,xx0],[yy0,yy0],color=gray,lw=.8,ls=':')
ax.scatter([xx0],[yy0],s=44,facecolors='white',edgecolors=green,lw=1.5,zorder=5)
ax.annotate(r'$\theta=\pi/3$'+'\n'+r'$(X,Y)=(1/2,-1/2)$',(xx0,yy0),xytext=(.31,.85),textcoords='axes fraction',fontsize=11,color='#555E66',arrowprops={'arrowstyle':'-','color':gray,'lw':.7})
ax.axhline(0,color=gray,lw=.7);ax.axvline(0,color=gray,lw=.7)
ax.set_xlim(-1.12,1.12);ax.set_ylim(-1.18,1.18);ax.set_xticks([-1,0,1]);ax.set_yticks([-1,0,1]);ax.set_xlabel(r'$X$',labelpad=9);ax.set_ylabel(r'$Y$',labelpad=8)
ax.set_title('(b) Deterministic dependence',loc='left',pad=17,fontsize=13)
ax.text(.5,-.28,r'$Y=2X^2-1$',transform=ax.transAxes,ha='center',fontsize=16,color=green)
for ax in axs:
    ax.spines['left'].set_color(gray);ax.spines['bottom'].set_color(gray);ax.tick_params(length=3,color=gray)
fig.text(.52,.125,r'$\theta\sim\mathcal{U}(0,2\pi):\quad\mathbb{E}[X]=\mathbb{E}[Y]=\mathbb{E}[XY]=0$',ha='center',fontsize=15)
fig.text(.52,.045,'Zero covariance does not imply independence.',ha='center',fontsize=15)
stem=OUT/'figura_ortogonalidade_dependencia'
for suffix in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+suffix),dpi=450,facecolor='white')
fig.savefig(OUT/'previa.png',dpi=150,facecolor='white');plt.close(fig)
(OUT/'legenda.txt').write_text('Ortogonalidade e dependência entre harmônicos. (a) X = cos θ e Y = cos 2θ em um período. Para fase uniforme em [0,2π), ambas as variáveis têm média zero e E[XY] = 0; são, portanto, não correlacionadas. (b) Os mesmos pares (X,Y) pertencem exatamente à parábola Y = 2X² − 1. O ponto destacado corresponde à fase π/3 nos dois painéis. Assim, Y é completamente determinada por X e as variáveis não são independentes. A figura é analítica e não representa uma simulação de separação. A ortogonalidade temporal não estabelece, por si só, independência estatística.',encoding='utf-8')
for p,h in hashes.items():assert hashlib.sha256(p.read_bytes()).digest()==h
print(f'Identidades verificadas; {len(hashes)} arquivos Word preservados.')

