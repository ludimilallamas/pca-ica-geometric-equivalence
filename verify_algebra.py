from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'generated/algebra';FIG=OUT/'figuras';FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
lam=np.array([12.,5.,3.,2.]);r=3
kappa=np.array([lam[j+1:].mean() for j in range(r)])
corrected=lam[:r]-kappa;c=(kappa.max()+kappa.min())/2
gap=(kappa.max()-kappa.min())/2
assert np.allclose(kappa[:-1]-kappa[1:],corrected[1:]/np.arange(3,1,-1))
assert np.isclose(np.max(abs(kappa-c)),gap)
rng=np.random.default_rng(2026112303)
x=rng.normal(size=80);L=6
Y=np.lib.stride_tricks.sliding_window_view(x,L)[:,::-1].T;K=Y.shape[1]
U,s,Vt=np.linalg.svd(Y,full_matrices=False);r=3
U=U[:,:r];V=Vt[:r,:].T;values=s[:r]**2/K
allvals=np.r_[s**2/K,np.zeros(K-len(s))]
kap=np.array([allvals[j+1:].mean() for j in range(r)])
tilde=values-kap;assert np.all(tilde>0)
Ht=Y@V/np.sqrt(K*tilde)
St=np.sqrt(K*tilde)[:,None]*V.T
Pr=U@U.T
err_pair=np.linalg.norm(Ht@St-Pr@Y)
Q=np.linalg.qr(rng.normal(size=(r,r)))[0]
W=np.diag(1/np.sqrt(values))@U.T;B=Q@W
Bplus=U@np.diag(np.sqrt(values))@Q.T
err_all=np.linalg.norm(Bplus@B-Pr)
assert err_pair<1e-11 and err_all<1e-11
theta=np.linspace(0,np.pi/2,181);M=np.diag([1.,0.]);D=M.copy()
errors=[]
for th in theta:
    Q2=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    T=np.diag([3.,2.]);A=T@Q2.T@M@Q2@np.linalg.inv(T)
    errors.append(np.linalg.norm(A-D,2))
assert errors[0]<1e-12 and errors[-1]>.99
# Igualdade no subespaco observado e desigualdade em entradas novas.
Yr=np.array([[1.,2.],[0.,0.]])
f=np.array([1.,1.]);g=np.array([1.,0.])
assert np.allclose(f@Yr,g@Yr) and not np.allclose(f,g)

fig,axs=plt.subplots(1,2,figsize=(10.6,4.1))
axs[0].plot([1,2,3],kappa,'o-',label='Correção NR: κj')
axs[0].axhline(c,color='#d87934',ls='--',label='Melhor piso comum c*')
axs[0].set(xlabel='Componente j',ylabel='Valor subtraído',xticks=[1,2,3],title='NR não é um deslocamento único neste exemplo')
axs[0].legend(fontsize=8)
axs[1].bar([1,2,3],lam[:3],width=.36,label='Autovalor observado')
axs[1].bar(np.arange(1,4)+.36,corrected,width=.36,label='Autovalor NR')
axs[1].set(xlabel='Componente j',xticks=[1.18,2.18,3.18],xticklabels=['1','2','3'],title='Autovalores hipotéticos 12, 5, 3, 2')
axs[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(FIG/'06_pisos_NR.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10.6,4.0))
axs[0].plot(theta*180/np.pi,errors,color='#265a88')
axs[0].set(xlabel='Rotação Q em graus',ylabel='‖AICA − APCA‖₂',title='Seleção de uma componente',xticks=[0,30,60,90])
axs[1].axis('off')
for y,title,body in [(.85,'Todas as r componentes','B⁺ B = Ur Urᵀ para qualquer Q ortogonal'),(.53,'Apenas um grupo','Igualdade exige Qᵀ M Q = D'),(.21,'Uma linha FIR de Ferreira','Igualdade exige o mesmo filtro na saída escalar')]:
    axs[1].text(.02,y,title,weight='bold',fontsize=11)
    axs[1].text(.02,y-.12,body,fontsize=10)
fig.tight_layout();fig.savefig(FIG/'07_equivalencia_operadores.png',dpi=180);plt.close(fig)
result={'semente':2026112303,'matriz_de_atrasos':{'T':80,'L':L,'K':K,'r':r},'kappa_exemplo':kappa.tolist(),'autovalores_NR_exemplo':corrected.tolist(),'melhor_piso_comum':float(c),'distancia_minima_ao_piso_comum':float(gap),'erro_sintese_NR_pareada':float(err_pair),'erro_reconstrucao_ICA_todas_componentes':float(err_all),'erro_selecao_theta0':float(errors[0]),'erro_selecao_theta90':float(errors[-1]),'escopo':'verificacao algebrica em precisao dupla; Q prescrito, sem executar ou reproduzir SOBI/Ferreira'}
(OUT/'verificacao_equivalencia_v3.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
