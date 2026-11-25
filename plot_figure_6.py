from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE=Path(__file__).resolve().parent
OUT=BASE/'generated/figure_6';OUT.mkdir(parents=True,exist_ok=True)
hashes={p:hashlib.sha256(p.read_bytes()).digest() for p in BASE.rglob('*.docx')}
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':12,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
blue='#244E73';orange='#AC5D2B';gray='#777E85'
lam=np.array([12.,5.,3.,2.]);kappa=np.array([lam[j+1:].mean() for j in range(3)])
c=(kappa.max()+kappa.min())/2;corrected=lam[:3]-kappa;err=(kappa.max()-kappa.min())/2
assert np.allclose(kappa,[10/3,2.5,2]) and np.isclose(err,2/3)
fig,axs=plt.subplots(1,2,figsize=(9,4.9),gridspec_kw={'width_ratios':[1.2,1]})
fig.subplots_adjust(left=.085,right=.96,top=.83,bottom=.30,wspace=.38)
y=np.array([3,2,1])
ax=axs[0]
for j in range(3):
    ax.annotate('',xy=(corrected[j],y[j]),xytext=(lam[j],y[j]),arrowprops={'arrowstyle':'-|>','lw':1.35,'color':gray,'shrinkA':8,'shrinkB':8,'mutation_scale':12})
    ax.scatter(lam[j],y[j],s=43,color=blue,zorder=4)
    ax.scatter(corrected[j],y[j],s=43,facecolors='white',edgecolors=orange,lw=1.5,zorder=4)
    ax.text((lam[j]+corrected[j])/2,y[j]+.18,[r'$-10/3$',r'$-5/2$',r'$-2$'][j],ha='center',fontsize=12,color='#454B50')
    ax.text(lam[j],y[j]-.23,str(int(lam[j])),ha='center',color=blue,fontsize=11)
    ax.text(corrected[j],y[j]-.23,[r'$26/3$',r'$5/2$',r'$1$'][j],ha='center',color=orange,fontsize=11)
ax.set_xlim(0,13);ax.set_ylim(.5,3.5);ax.set_yticks(y,['PC1','PC2','PC3']);ax.set_xticks([0,3,6,9,12]);ax.set_xlabel('Eigenvalue',labelpad=9)
ax.set_title('(a) Correction of each eigenvalue',loc='left',pad=18,fontsize=13)
ax.legend(handles=[Line2D([],[],ls='',marker='o',color=blue,label='Observed'),Line2D([],[],ls='',marker='o',mfc='white',mec=orange,mew=1.5,label='After NR')],loc='lower left',bbox_to_anchor=(-.03,-.35),ncol=2,frameon=False,fontsize=11,handletextpad=.3,columnspacing=1)

ax=axs[1]
ax.axvspan(c-err,c+err,color=blue,alpha=.055,zorder=0)
ax.axvline(c,color=gray,lw=1,ls=(0,(4,3)))
for yy,k in zip(y,kappa):
    ax.plot([c,k],[yy,yy],color=orange,lw=1.5)
    ax.scatter(k,yy,s=43,marker='s',facecolors='white',edgecolors=orange,lw=1.5,zorder=3)
for yy,k,label in zip(y,kappa,[r'$10/3$',r'$5/2$',r'$2$']):
    ax.text(k,yy+.21,label,ha='center',fontsize=12,color=orange)
ax.set_xlim(1.75,3.58);ax.set_ylim(.5,3.5);ax.set_yticks(y,['PC1','PC2','PC3']);ax.set_xticks([2,c,10/3],['2','8/3','10/3']);ax.set_xlabel('Subtracted value $\\kappa_j$',labelpad=9)
ax.set_title('(b) Best common shift',loc='left',pad=18,fontsize=13)
ax.text(.5,-.30,'$c^\\star=8/3\\approx2.667$',transform=ax.transAxes,ha='center',fontsize=12,color=gray)
for ax in axs:
    ax.tick_params(length=3,color=gray);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#ADB2B6');ax.tick_params(axis='y',length=0,pad=8)
fig.text(.5,.955,'NR correction: the subtracted value depends on the component',ha='center',va='top',fontsize=15)
fig.text(.5,.105,r'$\kappa_j=\frac{1}{4-j}\sum_{i=j+1}^{4}\widehat{\lambda}_i\qquad\qquad\min_c\max_j|\kappa_j-c|=\frac{2}{3}$',ha='center',fontsize=15)
fig.text(.5,.035,'Algebraic example · observed eigenvalues (12, 5, 3, 2) · three retained components',ha='center',fontsize=10.5,color='#555D64')
stem=OUT/'figura_pisos_nr'
for suffix in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+suffix),dpi=450,facecolor='white')
fig.savefig(OUT/'previa.png',dpi=150,facecolor='white');plt.close(fig)
(OUT/'legenda.txt').write_text('Comparação entre correção NR e deslocamento comum. (a) Cada seta liga um autovalor observado à sua versão corrigida, indicando a quantidade subtraída. (b) Os quadrados representam as correções individuais; os segmentos horizontais mostram seus desvios em relação à melhor aproximação comum c* = 8/3. A faixa sombreada vai de 2 a 10/3. Para os autovalores hipotéticos (12, 5, 3, 2), as correções são (10/3, 5/2, 2) e o erro mínimo máximo é 2/3. Na base ortonormal das três componentes retidas, esse é também o erro mínimo em norma de operador entre a correção diagonal NR e um múltiplo da identidade. Trata-se de um exemplo finito, sem implicar que aproximações por um piso comum sejam impossíveis em outros regimes.',encoding='utf-8')
for p,h in hashes.items():assert hashlib.sha256(p.read_bytes()).digest()==h
print(f'Figura gerada; {len(hashes)} arquivos Word preservados. Valores e erro verificados.')


