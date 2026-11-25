"""Experimentos ilustrativos; nao executa nem compara separadores ICA/SOBI."""
from pathlib import Path
import json, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent/'generated/noise_covariance'
FIG=OUT/'figuras'; FIG.mkdir(parents=True,exist_ok=True)
SEED=20261123; REPS=200; L=12
rng=np.random.default_rng(SEED)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#faf9f6','axes.facecolor':'#faf9f6'})
blue,green,orange='#265a88','#31856c','#d87934'

# Diagrama autoral, nao reproduz figura publicada.
fig,ax=plt.subplots(figsize=(11,5));ax.axis('off')
boxes=[(.02,.59,.27,.25,'Registro escalar x[t]\nImersão por atrasos\nY: L atrasos × K janelas'),(.365,.59,.27,.25,'Matrizes de segunda ordem\nR(0): covariância\nR(ℓ): covariâncias defasadas'),(.71,.59,.27,.25,'Separação e reconstrução\nSOBI e seleção de filtros\nObjetivo: fontes ou eventos')]
for x,y,w,h,s in boxes:
 ax.text(x+w/2,y+h/2,s,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.9',fc='#e5eeea',ec=green),fontsize=11)
for x in [.30,.645]:ax.annotate('',xy=(x+.055,.715),xytext=(x,.715),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=2,color=blue))
ax.text(.5,.19,'PCA de alta dimensão: corrigir autovalores/escores\nConexão possível na estimação espectral; equivalência completa não demonstrada',ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round,pad=1',fc='#edf2f4',ec=blue),fontsize=12)
ax.annotate('',xy=(.5,.49),xytext=(.5,.31),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=2,color=orange))
fig.tight_layout();fig.savefig(FIG/'01_conexao.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(11,3.9))
for ax,lag in zip(axs,[0,1,3]):
 C=np.eye(7,k=-lag)
 ax.imshow(C,cmap='Blues',vmin=0,vmax=1)
 ax.set(title=f'Covariância do ruído: atraso ℓ = {lag}',xlabel='Coordenada j',ylabel='Coordenada i',xticks=range(7),yticks=range(7))
fig.text(.5,.01,'Ruído branco escalar com variância 1, após imersão. Para |ℓ| ≥ L, a matriz populacional é zero.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1));fig.savefig(FIG/'02_ruido_defasado.png',dpi=170);plt.close(fig)

rows=[]
for T in [256,1024,4096]:
 records=[]
 vals={lag:[] for lag in [0,1,3,12]};norms={lag:[] for lag in vals}
 for _ in range(REPS):
  e=rng.normal(size=T)
  records.append(e.copy())
  Y=np.lib.stride_tricks.sliding_window_view(e,L)[:,::-1].T
  K=Y.shape[1]
  for lag in vals:
   C=Y@Y.T/K if lag==0 else Y[:,lag:]@Y[:,:-lag].T/(K-lag)
   target=np.eye(L,k=-lag)
   vals[lag].append(float(np.linalg.norm(C-target,'fro')/np.sqrt(L)))
   norms[lag].append(float(np.linalg.norm(C,'fro')/np.sqrt(L)))
 np.savez_compressed(OUT/f'noise_records_T{T}.npz', records=np.array(records), T=T, L=L, seed=SEED)
 for lag,v in vals.items():
  rows.append(dict(T=T,L=L,K=T-L+1,lag=lag,repeticoes=REPS,erro_medio=float(np.mean(v)),erro_padrao_monte_carlo=float(np.std(v,ddof=1)/np.sqrt(REPS)),norma_media=float(np.mean(norms[lag])),norma_alvo=float(np.linalg.norm(np.eye(L,k=-lag),'fro')/np.sqrt(L))))
with (OUT/'resultados_monte_carlo.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

fig,axs=plt.subplots(1,2,figsize=(11,4.7))
for lag,c in zip([0,1,3,12],[blue,green,orange,'#8a6386']):
 sub=[r for r in rows if r['lag']==lag]
 ts=[r['T'] for r in sub]
 axs[0].errorbar(ts,[r['erro_medio'] for r in sub],yerr=[1.96*r['erro_padrao_monte_carlo'] for r in sub],marker='o',color=c,label=f'ℓ = {lag}',capsize=4)
 axs[1].plot(ts,[r['norma_media'] for r in sub],marker='o',color=c,label=f'ℓ = {lag}')
 axs[1].axhline(sub[0]['norma_alvo'],color=c,ls='--',alpha=.5,lw=1)
for ax in axs:
 ax.set_xscale('log',base=2);ax.set_xticks([256,1024,4096],['256','1.024','4.096']);ax.set_xlabel('Comprimento do registro T');ax.legend(fontsize=9)
axs[0].set(title='Erro até a matriz populacional correta',ylabel='Média de ‖R estimada − R alvo‖F / √L')
axs[1].set(title='Covariância defasada não desaparece',ylabel='Média de ‖R estimada‖F / √L')
fig.text(.5,.01,'L = 12; 200 realizações por tamanho. Barras: ±1,96 erros-padrão Monte Carlo. Tracejados: normas populacionais.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1));fig.savefig(FIG/'03_monte_carlo.png',dpi=170);plt.close(fig)

theta=np.linspace(0,2*np.pi,1200,endpoint=False)
fig,axs=plt.subplots(1,2,figsize=(10,4.3))
axs[0].plot(theta,np.cos(theta),label='X = cos θ',color=blue);axs[0].plot(theta,np.cos(2*theta),label='Y = cos 2θ',color=orange)
axs[0].set(title='Harmônicos ortogonais em um período',xlabel='Fase θ',ylabel='Amplitude');axs[0].legend(fontsize=9)
axs[1].plot(np.cos(theta),np.cos(2*theta),color=green,lw=2)
axs[1].set(title='Dependência determinística: Y = 2X² − 1',xlabel='X',ylabel='Y')
fig.text(.5,.01,'Para θ uniforme: E[X] = E[Y] = E[XY] = 0. A covariância zero não implica independência.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1));fig.savefig(FIG/'04_ortogonalidade.png',dpi=170);plt.close(fig)

hat=np.array([12.,5.,3.,2.]);tilde=np.array([hat[j]-hat[j+1:].mean() for j in range(3)])
covnr=hat[:3]/tilde
assert np.all(tilde>0)
D=np.diag(np.sqrt(covnr));assert np.allclose(np.linalg.inv(D)@D,np.eye(3))
fig,ax=plt.subplots(figsize=(8,4.4));x=np.arange(3)
ax.bar(x-.23,np.ones(3),.23,label='Branqueamento usual',color=blue)
ax.bar(x,covnr,.23,label='Apenas substituir por λ corrigido',color=orange)
ax.bar(x+.23,np.ones(3),.23,label='Após rebranqueamento',color=green)
ax.set(xticks=x,xticklabels=['PC1','PC2','PC3'],ylabel='Variância amostral',title='Correção de escala e cancelamento pelo rebranqueamento',ylim=(0,4.1))
ax.legend(fontsize=8,loc='upper left');fig.text(.5,.01,'Exemplo algébrico: autovalores observados (12, 5, 3, 2). Não representa desempenho de um separador.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,1));fig.savefig(FIG/'05_rebranqueamento.png',dpi=170);plt.close(fig)

meta={'seed':SEED,'repeticoes':REPS,'L':L,'resultados':rows,'exemplo_autovalores':{'observados':hat.tolist(),'corrigidos':tilde.tolist(),'variancias_apos_substituicao':covnr.tolist()},'escopo':'Ilustracoes de identidades e covariancias. Nao constitui comparacao de desempenho ICA/SOBI/NR.'}
(OUT/'resultados.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(meta,ensure_ascii=False,indent=2))
