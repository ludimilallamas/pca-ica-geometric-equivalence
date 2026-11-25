from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator,FixedLocator,FixedFormatter
BASE=Path(__file__).resolve().parent
OUT=BASE/'generated/figure_8';OUT.mkdir(parents=True,exist_ok=True)
hashes={p:hashlib.sha256(p.read_bytes()).digest() for p in BASE.rglob('*.docx')}
source=BASE/'reference_sobi.json';data=json.loads(source.read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
blue='#244E73';orange='#AC5D2B';green='#397A68';gray='#899198'
fig=plt.figure(figsize=(9,7.9));gs=fig.add_gridspec(2,2,left=.10,right=.96,top=.93,bottom=.13,hspace=.74,wspace=.34,height_ratios=[1.05,1])
ax=fig.add_subplot(gs[0,:]);handles=[]
methods=[('longos','Lags 12 and 13',blue,'o'),('curtos','Lags 1 and 2 · uncorrected',orange,'^'),('curtos_corrigidos','Lags 1 and 2 · known $\\sigma^2$',green,'s')]
for method,label,color,marker in methods:
    sub=sorted([s for s in data['summaries'] if s['metodo']==method],key=lambda s:s['T'])
    for s in sub:
        vals=[r['erro_operadores'] for r in data['rows'] if r['T']==s['T'] and r['metodo']==method]
        assert len(vals)==40 and np.isclose(np.mean(vals),s['media'])
    h=ax.errorbar([s['T'] for s in sub],[s['media'] for s in sub],yerr=[1.96*s['EP_MC'] for s in sub],label=label,color=color,marker=marker,mfc='white',mew=1.2,ms=5,lw=1.5,capsize=3)
    handles.append(h)
ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(315,7500);ax.set_ylim(.010,.58)
ax.set_xticks([384,1536,6144],['384','1,536','6,144']);ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_locator(FixedLocator([.01,.03,.1,.3]));ax.yaxis.set_major_formatter(FixedFormatter(['0.01','0.03','0.10','0.30']))
ax.set_xlabel('Record length $T$',labelpad=6);ax.set_ylabel(r'$\|\widehat{A}_{\rm SOBI}-\widehat{P}_{\rm NR}\|_F/\sqrt{r}$',labelpad=8)
ax.set_title('(a) Difference between estimated operators',loc='left',fontsize=13,pad=15)
ax.grid(axis='y',color='#E5E8EA',lw=.6);ax.set_axisbelow(True)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.56),frameon=False,ncol=1,fontsize=10.5,handlelength=2.2,labelspacing=.45)
L=data['L'];ells=np.arange(25);profiles=[]
for k,a in zip(data['frequencias_grade'],data['amplitudes']):
    nu=L*a*a/4;rho=nu/(nu+data['variancia_ruido']);profiles.append(rho*np.cos(2*np.pi*k*ells/L))
ax=fig.add_subplot(gs[1,0]);ax.axvspan(11.7,13.3,color=blue,alpha=.08)
for k,values,color,marker in zip([1,3],profiles,[blue,green],['o','s']):
    ax.plot(ells,values,color=color,lw=1,marker=marker,ms=2.8,label=rf'$k={k}$')
    ax.scatter([12,13],values[[12,13]],s=28,marker=marker,facecolors='white',edgecolors=color,lw=1.3,zorder=4)
ax.axhline(0,color=gray,lw=.65);ax.set_xlim(-.5,24.5);ax.set_ylim(-1,1.04);ax.set_xticks([0,6,12,18,24]);ax.set_yticks([-1,0,1]);ax.set_xlabel('Lag $\\ell$',labelpad=7);ax.set_ylabel(r'$\rho_k\cos(2\pi k\ell/L)$',labelpad=7)
ax.set_title('(b) Temporal profiles of the blocks',loc='left',fontsize=12,pad=15)
ax=fig.add_subplot(gs[1,1]);ax.set_xlim(-.08,1.03);ax.set_ylim(-.10,.95)
for k,values,color,marker in zip([1,3],profiles,[blue,green],['o','s']):
    x,y=values[12],values[13];ax.plot([x,x],[0,y],color=color,ls=':',lw=.8);ax.plot([0,x],[y,y],color=color,ls=':',lw=.8)
    ax.scatter(x,y,s=65,marker=marker,facecolors='white',edgecolors=color,lw=1.7,zorder=4)
    ax.annotate(rf'$k={k}$',(x,y),xytext=(7,7),textcoords='offset points',color=color,fontsize=13)
ax.axhline(0,color=gray,lw=.7);ax.axvline(0,color=gray,lw=.7)
ax.set_xticks([0,.5,1],['0','0.5','1']);ax.set_yticks([0,.5],['0','0.5'])
ax.set_xlabel('Signature at $\\ell=L=12$',labelpad=7);ax.set_ylabel('Signature at $\\ell=L+1=13$',labelpad=7)
ax.set_title('(c) Signatures at two lags',loc='left',fontsize=12,pad=15)
fig.text(.54,.027,'$L=12$ · $\\sigma^2=2$ · $r=2$ · 40 records per length · bars: $\\pm1.96$ Monte Carlo standard errors',ha='center',fontsize=10,color='#555E66')
for ax in fig.axes:
    ax.spines['left'].set_color(gray);ax.spines['bottom'].set_color(gray);ax.tick_params(length=3,color=gray)
stem=OUT/'figura_sobi_defasagens'
for suffix in ['png','svg','pdf']:fig.savefig(stem.with_suffix('.'+suffix),dpi=450,facecolor='white')
fig.savefig(OUT/'previa.png',dpi=150,facecolor='white');plt.close(fig)
(OUT/'legenda.txt').write_text('Verificação das hipóteses com SOBI estimado. (a) Diferença normalizada entre os operadores SOBI e geométrico NR, preservando integralmente as médias e barras de ±1,96 Monte Carlo standard errors do experimento original, com 40 records per length. Ambos os eixos são logarítmicos. A variante curta corrigida utiliza a variância verdadeira do ruído e constitui uma referência com parâmetro conhecido. (b) Assinaturas populacionais após branqueamento dos blocos k = 1 e k = 3, com L = 12; a faixa destaca as defasagens 12 e 13. (c) Cada bloco é representado pelo par de assinaturas nessas defasagens. Os pontos distintos ilustram a identificação dos blocos neste modelo harmônico, sem determinar uma orientação única dentro de cada bloco. As rotações SOBI do painel (a) foram estimadas; não foram prescritas. Não foram feitas novas simulações: o redesenho usa resultados_sobi_v6.json.',encoding='utf-8')
for p,h in hashes.items():assert hashlib.sha256(p.read_bytes()).digest()==h
print(f'Resultados originais conferidos. {len(hashes)} arquivos Word preservados.')
