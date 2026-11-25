from pathlib import Path
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE=Path(__file__).resolve().parent
OUT=BASE/'generated/figure_7';OUT.mkdir(parents=True,exist_ok=True)
hashes={p:hashlib.sha256(p.read_bytes()).digest() for p in BASE.rglob('*.docx')}
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
blue='#244E73';orange='#AC5D2B';gray='#899198'
fig=plt.figure(figsize=(9,7.6));gs=fig.add_gridspec(2,2,left=.10,right=.96,bottom=.23,top=.92,hspace=.65,height_ratios=[1.1,1],wspace=.35)
angle=np.deg2rad(40);Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
z=np.array([.85,.65]);p=np.array([z[0],0]);v=Q[0];q=v*(v@z)
t=np.linspace(0,2*np.pi,400)
def setup(ax):
    ax.set_aspect('equal');ax.set_xlim(-1.25,1.25);ax.set_ylim(-1.25,1.25)
    for s in ['left','bottom']:
        ax.spines[s].set_position('zero');ax.spines[s].set_color('#B2B8BD');ax.spines[s].set_linewidth(.7)
    ax.set_xticks([]);ax.set_yticks([])
    ax.plot(np.cos(t),np.sin(t),color=gray,lw=.8,ls=(0,(3,3)))
def arrow(ax,point,color,lw=1.5):
    ax.annotate('',xy=point,xytext=(0,0),arrowprops={'arrowstyle':'-|>','lw':lw,'color':color,'mutation_scale':11})
ax=fig.add_subplot(gs[0,0]);setup(ax)
for k in range(2):
    vv=Q[k];ax.plot([-vv[0],vv[0]],[-vv[1],vv[1]],color=orange,lw=1,alpha=.8)
    ax.text(*(vv*1.12),rf'$q_{k+1}$',color=orange,ha='center',va='center',fontsize=12)
    comp=vv*(vv@z);arrow(ax,comp,orange,1)
    ax.plot([comp[0],z[0]],[comp[1],z[1]],color=orange,lw=.7,ls=':',alpha=.6)
arrow(ax,z,blue);ax.scatter(*z,s=26,c=blue,zorder=5);ax.text(z[0]+.08,z[1]+.03,r'$z$',color=blue,fontsize=14)
ax.set_title('(a) All components: same reconstruction',fontsize=12,pad=17)
ax.text(.5,-.16,r'$Q^{\mathsf{T}}Qz=z$',transform=ax.transAxes,ha='center',fontsize=14)
ax=fig.add_subplot(gs[0,1]);setup(ax)
ax.plot([-v[0],v[0]],[-v[1],v[1]],color=orange,lw=.9)
ax.plot([0,z[0]],[0,0],color=blue,lw=2)
ax.plot([p[0],z[0]],[p[1],z[1]],color=blue,lw=.8,ls=':')
ax.plot([q[0],z[0]],[q[1],z[1]],color=orange,lw=.8,ls=':')
arrow(ax,p,blue);arrow(ax,q,orange)
ax.scatter(*z,s=25,c='#555E66',zorder=5);ax.text(z[0]+.08,z[1]+.03,r'$z$',fontsize=14,color='#555E66')
ax.scatter(*p,s=30,facecolors='white',edgecolors=blue,zorder=5)
ax.scatter(*q,s=30,marker='s',facecolors='white',edgecolors=orange,zorder=5)
ax.text(.92,-.18,r'$Dz$',color=blue,fontsize=13)
ax.text(q[0]-.07,q[1]-.25,r'$Q^{\mathsf{T}}MQz$',color=orange,fontsize=12,ha='center')
ax.set_title('(b) One component: selection may change',fontsize=12,pad=17)
ax.text(.5,-.16,r'$Q^{\mathsf{T}}MQ\ne D$',transform=ax.transAxes,ha='center',fontsize=14)

theta=np.linspace(0,np.pi/2,361);D=np.diag([1.,0.]);T=np.diag([3.,2.]);errors=[]
for th in theta:
    R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    A=T@R.T@D@R@np.linalg.inv(T);errors.append(np.linalg.norm(A-D,2))
assert errors[0]<1e-12 and np.isclose(errors[-1],1)
ax=fig.add_subplot(gs[1,:]);ax.plot(np.rad2deg(theta),errors,color=blue,lw=1.7)
for deg in [0,40,90]:
    ix=np.argmin(abs(np.rad2deg(theta)-deg));ax.scatter(deg,errors[ix],s=30,c=orange,zorder=3)
ax.axvline(40,color=gray,lw=.7,ls=(0,(3,3)))
ax.text(41,.10,'40°: example in panel (b)',fontsize=10,color='#59636B')
ax.set(xlim=(-2,92),ylim=(-.04,1.22),xticks=[0,30,40,60,90],yticks=[0,.5,1],xlabel='Prescribed rotation angle of $Q$ (degrees)',ylabel=r'$\|A_{\rm ICA}-A_{\rm PCA}\|_2$')
ax.set_title('(c) Operator difference in the original space',loc='left',fontsize=12,pad=14)
ax.tick_params(length=3,color=gray);ax.spines['left'].set_color(gray);ax.spines['bottom'].set_color(gray)
fig.text(.53,.134,r'$\Lambda=\operatorname{diag}(9,4),\quad M=D=\operatorname{diag}(1,0)$',ha='center',fontsize=12)
fig.text(.53,.075,'Selection equivalence: $Q^{\\mathsf{T}}MQ=D$',ha='center',fontsize=13)
fig.text(.53,.025,'FIR output: $\\eta b_i^{\\mathsf{T}}=e_a^{\\mathsf{T}}A_{\\rm PCA}$  ·  requires matching coefficients after gain adjustment',ha='center',fontsize=11,color='#555E66')
stem=OUT/'figura_equivalencia_operadores'
for suffix in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+suffix),dpi=450,facecolor='white')
fig.savefig(OUT/'previa.png',dpi=150,facecolor='white');plt.close(fig)
(OUT/'legenda.txt').write_text('Equivalência e limites na reconstrução. (a,b) Geometria no plano branqueado: conservar ambas as componentes preserva z; conservar apenas a primeira direção PCA ou a primeira direção rotacionada pode produzir projeções diferentes. A rotação prescrita é de 40 graus. (c) Norma de operador da diferença no espaço original, com U = I, Λ = diag(9,4), M = D = diag(1,0), A_ICA = Λ^(1/2) Qᵀ M Q Λ^(-1/2) e A_PCA = D. Os painéis superiores representam o espaço branqueado; o gráfico inferior inclui a transformação inversa do branqueamento, portanto não deve ser interpretado como simples distância entre as duas projeções superiores. A igualdade exige Qᵀ M Q = D. Para a saída FIR, exige-se ainda igualdade dos coeficientes após seleção e ganho. Q é prescrito para ilustrar a álgebra; não foi estimado por SOBI. O exemplo não mede desempenho de remoção de ruído.',encoding='utf-8')
for p,h in hashes.items():assert hashlib.sha256(p.read_bytes()).digest()==h
print(f'Figura gerada; {len(hashes)} arquivos Word preservados; identidades verificadas.')
