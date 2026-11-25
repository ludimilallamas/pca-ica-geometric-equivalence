"""Figura autônoma. Não abre nem altera nenhum documento Word.
Os painéis de dados são simulações novas e reproduzíveis.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
BASE=ROOT
OUT=BASE/'generated/figure_1';OUT.mkdir(parents=True,exist_ok=True)
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in BASE.rglob('*.docx')}
SEED=2026112308;rng=np.random.default_rng(SEED)
BLUE='#244E73';ORANGE='#AC5D2B';GRAY='#8B9298';INK='#24292E'
plt.rcParams.update({'font.family':'STIXGeneral','font.size':12,'mathtext.fontset':'stix','axes.labelsize':11,'xtick.labelsize':10,'ytick.labelsize':10,'axes.linewidth':.65,'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42})

def dual_points(d,reps=32):
    points=[[],[]]
    for _ in range(reps):
        X=rng.normal(size=(d,2))
        G=X.T@X/d
        vals,U=np.linalg.eigh(G);order=np.argsort(vals)[::-1]
        for j,k in enumerate(order):
            w=vals[k]*U[:,k];points[j].extend([w,-w])
    return [np.array(p) for p in points]

def sobi2(Cs):
    # A rotação ótima para duas coordenadas minimiza uma forma quadrática em 2θ.
    b=np.array([C[0,1] for C in Cs]);g=np.array([(C[1,1]-C[0,0])/2 for C in Cs])
    G=np.array([[b@b,b@g],[b@g,g@g]])
    _,V=np.linalg.eigh(G);v=V[:,0];theta=np.arctan2(v[1],v[0])/2
    c,s=np.cos(theta),np.sin(theta);Q=np.array([[c,s],[-s,c]])
    order=np.argsort(np.diag(Q@Cs[0]@Q.T))[::-1]
    return Q[order]

def ar1(phi,T):
    a=np.zeros(T);a[0]=rng.normal();e=rng.normal(scale=np.sqrt(1-phi*phi),size=T-1)
    for t in range(1,T):a[t]=phi*a[t-1]+e[t-1]
    return a

def sobi_directions(reps=32,T=512):
    pts=[[],[]];residuals=[]
    for _ in range(reps):
        x=ar1(.9,T)+.7*ar1(-.35,T)+.35*rng.normal(size=T)
        Y=np.vstack([x[1:],x[:-1]]);Y=Y-Y.mean(axis=1,keepdims=True)
        K=Y.shape[1];C0=Y@Y.T/K
        ev,U=np.linalg.eigh(C0);W=(U*(1/np.sqrt(ev)))@U.T
        Z=W@Y;assert np.allclose(Z@Z.T/K,np.eye(2),atol=1e-10)
        Cs=[]
        for lag in [1,2]:
            C=Z[:,lag:]@Z[:,:-lag].T/(K-lag);Cs.append((C+C.T)/2)
        Q=sobi2(Cs)
        for j,q in enumerate(Q):pts[j].extend([q,-q])
        residuals.append(sum(np.sum((Q@C@Q.T-np.diag(np.diag(Q@C@Q.T)))**2) for C in Cs))
    return [np.array(a) for a in pts],residuals

low=dual_points(20);high=dual_points(2000);sd,residuals=sobi_directions()
fig,axs=plt.subplots(2,2,figsize=(8.4,7.3))
fig.subplots_adjust(left=.08,right=.97,bottom=.19,top=.935,wspace=.22,hspace=.44)
ang=np.linspace(0,2*np.pi,500)

def axes(ax,limit=1.65):
    ax.set_aspect('equal');ax.set_xlim(-limit,limit);ax.set_ylim(-limit,limit)
    for spine in ['top','right']:ax.spines[spine].set_visible(False)
    ax.spines['left'].set_position('zero');ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_color('#9A9EA3');ax.spines['bottom'].set_color('#9A9EA3')
    ax.set_xticks([-1,1]);ax.set_yticks([-1,1]);ax.tick_params(length=3,pad=4,color='#9A9EA3')
    ax.plot(np.cos(ang),np.sin(ang),color='#666D73',lw=.8,ls=(0,(3,3)),zorder=0)

def scatter(ax,points):
    ax.scatter(points[0][:,0],points[0][:,1],s=26,marker='o',facecolors='white',edgecolors=BLUE,linewidths=.85,zorder=3)
    ax.scatter(points[1][:,0],points[1][:,1],s=30,marker='^',facecolors='white',edgecolors=ORANGE,linewidths=.85,zorder=3)

maxlow=max(abs(x).max() for x in low);limit=max(1.65,np.ceil((maxlow+.15)*10)/10)
for ax,points,title in [(axs[0,0],low,'(a) Dual geometry · $d=20$'),(axs[0,1],high,'(b) Dual geometry · $d=2000$')]:
    axes(ax,limit);scatter(ax,points);ax.set_title(title,loc='left',fontsize=13,pad=13)
    ax.text(.5,-.13,r'$w_j=(2/d)S_D\widehat{u}_j$',transform=ax.transAxes,ha='center',va='top',fontsize=12)

ax=axs[1,0];axes(ax,1.65)
ax.set_title('(c) NR correction · radial dilation',loc='left',fontsize=12,pad=13)
theta=np.deg2rad(22);U=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
g=np.sqrt(np.array([12.,5.])/np.array([26/3,2.5]))
for j,color in enumerate([BLUE,ORANGE]):
    u=U[:,j];v=g[j]*u
    ax.plot([-v[0],v[0]],[-v[1],v[1]],color=color,lw=.7,alpha=.4)
    ax.annotate('',xy=v,xytext=(0,0),arrowprops={'arrowstyle':'-|>','mutation_scale':12,'lw':1.5,'color':color},zorder=4)
    ax.scatter([u[0],-u[0]],[u[1],-u[1]],s=30,facecolors='white',edgecolors=GRAY,lw=1,zorder=5)
    ax.scatter([v[0],-v[0]],[v[1],-v[1]],s=22,marker='o' if j==0 else '^',color=color,zorder=5)
    ax.text(v[0]+(.1 if j==0 else -.32),v[1]+(.06 if j==0 else .07),rf'$\widetilde{{h}}_{j+1}$',fontsize=13,color=color)
ax.text(.5,-.13,r'$\widetilde{h}_j=\sqrt{\widehat{\lambda}_j/\widetilde{\lambda}_j}\,\widehat{h}_j$',transform=ax.transAxes,ha='center',va='top',fontsize=13)

ax=axs[1,1];axes(ax,1.65)
ax.set_title('(d) SOBI · lag-based orientation',loc='left',fontsize=12,pad=13)
scatter(ax,sd)
for j,color in enumerate([BLUE,ORANGE]):
    A=sd[j][::2];C=A.T@A;_,V=np.linalg.eigh(C);v=V[:,-1]
    if v[1]<0:v=-v
    ax.annotate('',xy=v*.87,xytext=(0,0),arrowprops={'arrowstyle':'-|>','mutation_scale':10,'color':color,'lw':1.2})
ax.text(.5,-.13,r'$Q\widehat{R}_z(\ell)Q^{\mathsf{T}}\ \approx\ \mathrm{diagonal}$',transform=ax.transAxes,ha='center',va='top',fontsize=12)

# Legenda global curta, separada das equações e dos painéis.
from matplotlib.lines import Line2D
handles=[Line2D([],[],ls='',marker='o',mfc='white',mec=BLUE,label='First direction'),Line2D([],[],ls='',marker='^',mfc='white',mec=ORANGE,label='Second direction'),Line2D([],[],color='#666D73',ls=(0,(3,3)),lw=.8,label='Unit circle')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.006),ncol=3,frameon=False,fontsize=11,handlelength=1.7,columnspacing=1.8)
stem=OUT/'figura_geometria_pca_nr_sobi'
fig.savefig(stem.with_suffix('.png'),dpi=450)
fig.savefig(stem.with_suffix('.svg'))
fig.savefig(stem.with_suffix('.pdf'))
plt.close(fig)

caption=('Figura — Dual geometry, reescala espectral e orientação temporal. '
'(a,b) Simulações gaussianas novas e independentes, com n = 2, covariância I_d e 32 realizações por dimensão; mostram-se os pares ±w_j. '
'A aproximação da circunferência quando d aumenta ilustra a concentração da matriz dual normalizada, inspirada na linguagem gráfica da Figura 1 de Yata e Aoshima (2012). '
'(c) Dilatação radial de duas direções ilustrativas; os pontos cinzentos marcam vetores unitários e os coloridos suas versões reescaladas. '
'Os fatores são obtidos dos autovalores hipotéticos (12,5,3,2), cuja correção NR nas duas primeiras posições é (26/3,5/2). '
'(d) Direções unitárias estimadas por SOBI simétrico em 32 registros sintéticos de 512 amostras, com pseudocanais (x_t,x_{t-1}), branqueamento simétrico e defasagens 1 e 2. '
'O sinal escalar é a soma de dois AR(1) independentes e ruído branco. A concentração das direções é ilustrativa deste exemplo bidimensional; não representa prova de separação das fontes ou de equivalência geral entre PCA e ICA. '
'As coordenadas de cada painel pertencem ao objeto ali indicado, não a um mesmo sistema de dados. Os pontos do artigo original não foram reproduzidos.')
(OUT/'legenda_da_figura.txt').write_text(caption,encoding='utf-8')
metadata={'seed':SEED,'dual':{'n':2,'d':[20,2000],'realizacoes_por_d':32,'w':'(2/d) S_D uhat = (X.T X / d) uhat'},'NR':{'autovalores_observados':[12,5,3,2],'autovalores_corrigidos_primeiras_duas':[26/3,2.5],'fatores':g.tolist()},'SOBI':{'representacao':'y_t=(x_t,x_{t-1})','AR_phi':[.9,-.35],'pesos':[1,.7],'desvio_ruido':.35,'T':512,'repeticoes':32,'lags':[1,2],'residuos_JD':residuals},'figuras':['figura_geometria_pca_nr_sobi.png','figura_geometria_pca_nr_sobi.svg','figura_geometria_pca_nr_sobi.pdf'],'nenhum_word_alterado':True}
for path,digest in sources.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
(OUT/'parametros_e_verificacao.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'saida':str(stem),'limite_paineis_duais':limit,'arquivos_word_verificados_sem_alteracao':len(sources)},ensure_ascii=False))
