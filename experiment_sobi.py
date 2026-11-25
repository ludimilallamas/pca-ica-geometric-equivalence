"""SOBI simetrico por rotacoes de Jacobi; nao usa Q prescrito.
Modelo de harmônicos na grade, ruido branco escalar e janelas sobrepostas.
"""
from pathlib import Path
import json,csv,sys,time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'generated/sobi';FIG=OUT/'figuras';FIG.mkdir(parents=True,exist_ok=True)

def spectral(C):
    v,U=np.linalg.eigh((C+C.T)/2)
    if v.min()<=0:raise ValueError('Covariancia nao positiva')
    return (U*np.sqrt(v))@U.T,(U*(1/np.sqrt(v)))@U.T

def off(A):return A-np.diag(np.diag(A))

def sobi(Cs,max_sweeps=100,tol=1e-12):
    A=np.array(Cs,copy=True);p=A.shape[1];Q=np.eye(p)
    before=sum(np.sum(off(C)**2) for C in A)
    for sweep in range(max_sweeps):
        for i in range(p-1):
            for j in range(i+1,p):
                b=A[:,i,j];g=(A[:,j,j]-A[:,i,i])/2
                G=np.array([[b@b,b@g],[b@g,g@g]])
                if np.linalg.norm(G)<1e-25:continue
                _,V=np.linalg.eigh(G);v=V[:,0]
                phi=np.arctan2(v[1],v[0]);theta=phi/2
                theta=(theta+np.pi/4)%(np.pi/2)-np.pi/4
                c,s=np.cos(theta),np.sin(theta)
                J=np.eye(p);J[np.ix_([i,j],[i,j])]=[[c,s],[-s,c]]
                A=J@A@J.T;Q=J@Q
        after=sum(np.sum(off(C)**2) for C in A)
        assert after<=before+1e-10*(1+before),(before,after)
        if before-after<=tol*(1+before):break
        before=after
    return Q,float(after),sweep+1

def harmonic_model(L,ks,amps,nv):
    cols=[];nus=[]
    for k,a in zip(ks,amps):
        omega=2*np.pi*k/L;ii=np.arange(L)
        cols.extend([np.sqrt(2/L)*np.cos(omega*ii),np.sqrt(2/L)*np.sin(omega*ii)])
        nus.extend([L*a*a/4]*2)
    U=np.array(cols).T;assert np.allclose(U.T@U,np.eye(len(nus)))
    Ps=[U[:,2*j:2*j+2]@U[:,2*j:2*j+2].T for j in range(len(ks))]
    R0=sum(nus[2*j]*P for j,P in enumerate(Ps))+nv*np.eye(L)
    _,W=spectral(R0)
    def R(lag):return sum(nus[2*j]*np.cos(2*np.pi*ks[j]*lag/L)*P for j,P in enumerate(Ps))
    def C(lag):return W@R(lag)@W
    return R0,Ps,R,C,np.array(nus)

def run():
    seed=2026112306;rng=np.random.default_rng(seed)
    L=12;ks=[1,3];amps=[2.,1.];nv=2.;r=2
    R0,Ps,R,C,nus=harmonic_model(L,ks,amps,nv)
    S,W=spectral(R0);target=Ps[0]
    f=target[0,:]
    assert np.isclose(f@f,r/L)
    for phase in [.2,1.4,2.7]:
        samples=amps[0]*np.cos(phase-2*np.pi*ks[0]/L*np.arange(L))
        assert np.isclose(f@samples,amps[0]*np.cos(phase))
        nuisance=amps[1]*np.cos(phase-2*np.pi*ks[1]/L*np.arange(L))
        assert abs(f@nuisance)<1e-12
    Q,jpop,sw=sobi([C(L),C(L+1)])
    idx=np.argsort(np.diag(Q@C(L)@Q.T))[-r:]
    Pq=Q[idx,:].T@Q[idx,:];Apop=S@Pq@W
    errpop=float(np.linalg.norm(Apop-target,'fro'))
    assert jpop<1e-15 and errpop<1e-6,(jpop,errpop)
    # Covariância de ruído compartilhado não comuta, em geral, com os blocos.
    raw=[C(l)+nv*W@((np.eye(L,k=-l)+np.eye(L,k=l))/2)@W for l in [1,2]]
    comm=float(np.linalg.norm(raw[0]@raw[1]-raw[1]@raw[0],2))
    assert comm>1e-5
    # Verificação independente do limite determinístico da Proposição 11.
    tests=0;largest_ratio=0.
    delta=nus[0]-nus[2];rho=nus[0]/(nus[0]+nv)-nus[2]/(nus[2]+nv)
    for _ in range(60):
        E=rng.normal(size=(L,L));E=(E+E.T)*.001
        C0hat=R0+E;Sh,Wh=spectral(C0hat)
        Z=rng.normal(size=(L,L));Z=(Z+Z.T)*.0005
        H=C(L)+Z
        hv,V=np.linalg.eigh(H);P=V[:,-r:]@V[:,-r:].T
        cv,U=np.linalg.eigh(C0hat);Pg=U[:,-r:]@U[:,-r:].T
        e0=np.linalg.norm(E,2);eta=np.linalg.norm(Z,2)
        mh=cv[0];M=np.linalg.eigvalsh(R0)[-1];m=nv
        dS=e0/(np.sqrt(m)+np.sqrt(mh));dW=dS/np.sqrt(m*mh)
        bq=np.sqrt(2*r)*eta/(rho-eta)
        bp=np.sqrt(2*r)*e0/(delta-e0)
        bound=np.sqrt(cv[-1]/mh)*bq+np.sqrt(r)*(dS/np.sqrt(mh)+np.sqrt(M)*dW)+bp
        error=np.linalg.norm(Sh@P@Wh-Pg,'fro')
        assert error<=bound+1e-10,(error,bound)
        largest_ratio=max(largest_ratio,float(error/bound));tests+=1
    detuning_tests=0
    for _ in range(30):
        deviations=rng.uniform(-.005,.005,len(ks))
        for lag in [0,L,L+1]:
            h=lag+np.arange(L)[None,:]-np.arange(L)[:,None]
            Rpert=sum(a*a/2*np.cos((2*np.pi*k/L+dv)*h) for k,a,dv in zip(ks,amps,deviations))
            Rpert=(Rpert+Rpert.T)/2
            lim=L/2*sum(a*a*abs(dv) for a,dv in zip(amps,deviations))*(lag+L-1)
            assert np.linalg.norm(Rpert-R(lag),2)<=lim+1e-10
            detuning_tests+=1
    rows=[];REPS=40
    variants=[('longos',[L,L+1],False),('curtos',[1,2],False),('curtos_corrigidos',[1,2],True)]
    for T in [384,1536,6144]:
        signal_records=[]; noise_records=[]; phase_records=[]
        K=T-L+1
        for rep in range(REPS):
            phases=rng.uniform(0,2*np.pi,len(ks));tt=np.arange(T)
            sig=sum(a*np.cos(2*np.pi*k/L*tt+ph) for a,k,ph in zip(amps,ks,phases))
            x=sig+rng.normal(scale=np.sqrt(nv),size=T)
            signal_records.append(sig.copy()); noise_records.append((x-sig).copy()); phase_records.append(phases.copy())
            Y=np.lib.stride_tricks.sliding_window_view(x,L)[:,::-1].T
            Rhat0=Y@Y.T/K;Sh,Wh=spectral(Rhat0)
            cv,U=np.linalg.eigh(Rhat0);Pnr=U[:,-r:]@U[:,-r:].T
            cache={}
            for lag in [1,2,L,L+1]:
                Rhat=Y[:,lag:]@Y[:,:-lag].T/(K-lag)
                cache[lag]=(Rhat+Rhat.T)/2
            # Regra fixa de seleção: as r maiores diagonais do atraso L.
            selector=Wh@cache[L]@Wh
            for name,lags,corrected in variants:
                Cs=[]
                for lag in lags:
                    RR=cache[lag]
                    if corrected:RR=RR-nv*(np.eye(L,k=-lag)+np.eye(L,k=lag))/2
                    Cs.append(Wh@RR@Wh)
                Q,loss,sweeps=sobi(Cs)
                idx=np.argsort(np.diag(Q@selector@Q.T))[-r:]
                Pq=Q[idx,:].T@Q[idx,:];A=Sh@Pq@Wh
                row={'T':T,'K':K,'repeticao':rep,'metodo':name,'erro_operadores':float(np.linalg.norm(A-Pnr,'fro')/np.sqrt(r)),'erro_alvo':float(np.linalg.norm(A-target,'fro')/np.sqrt(r)),'residuo_JD':loss,'sweeps':sweeps}
                if name=='longos':
                    e0=np.linalg.norm(Rhat0-R0,2)
                    tau=np.linalg.norm(off(Q@selector@Q.T),2)
                    eta=np.linalg.norm(selector-C(L),2)+tau
                    if e0<delta/2 and eta<rho/2:
                        mh=cv[0];m=nv;M=nus[0]+nv
                        dS=e0/(np.sqrt(m)+np.sqrt(mh));dW=dS/np.sqrt(m*mh)
                        bound=np.sqrt(cv[-1]/mh)*np.sqrt(2*r)*eta/(rho-eta)+np.sqrt(r)*(dS/np.sqrt(mh)+np.sqrt(M)*dW)+np.sqrt(2*r)*e0/(delta-e0)
                        actual=np.linalg.norm(A-Pnr,'fro')
                        assert actual<=bound+1e-9,(actual,bound)
                        row['limite_normalizado']=float(bound/np.sqrt(r))
                rows.append(row)
        np.savez_compressed(OUT/f'harmonic_records_T{T}.npz', signal=np.array(signal_records), noise=np.array(noise_records), phases=np.array(phase_records), T=T, L=L, seed=seed)
        print('Concluido T =',T,flush=True)
    summaries=[]
    for T in [384,1536,6144]:
        for name,_,_ in variants:
            sub=[z for z in rows if z['T']==T and z['metodo']==name]
            vals=np.array([z['erro_operadores'] for z in sub])
            summaries.append({'T':T,'metodo':name,'media':float(vals.mean()),'EP_MC':float(vals.std(ddof=1)/np.sqrt(REPS)),'mediana':float(np.median(vals)),'JD_medio':float(np.mean([z['residuo_JD'] for z in sub])),'limites_verificados':sum('limite_normalizado' in z for z in sub)})
    data={'seed':seed,'repeticoes':REPS,'L':L,'frequencias_grade':ks,'amplitudes':amps,'variancia_ruido':nv,'posto_selecionado':r,'T':[384,1536,6144],'erro_populacional':errpop,'residuo_populacional':jpop,'comutador_lags_curtos':comm,'testes_limite':tests,'testes_desvio_frequencia':detuning_tests,'maior_razao_erro_limite':largest_ratio,'summaries':summaries,'rows':rows,'runtime':{'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__}}
    (OUT/'resultados_sobi_v6.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'resultados_sobi_v6.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(rows[-1]);fields=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for name,label,color in [('longos','Defasagens L e L+1','#265a88'),('curtos','Defasagens 1 e 2 sem correção','#d87934'),('curtos_corrigidos','Defasagens 1 e 2 com σ² conhecida','#31856c')]:
        sub=[s for s in summaries if s['metodo']==name]
        axs[0].errorbar([s['T'] for s in sub],[s['media'] for s in sub],yerr=[1.96*s['EP_MC'] for s in sub],label=label,color=color,marker='o',capsize=3)
    axs[0].set(xscale='log',yscale='log',xlabel='Comprimento do registro T',ylabel='‖A estimado SOBI − P estimado NR‖F / √r',title='Operadores estimados, sem Q prescrito')
    axs[0].set_xticks([384,1536,6144],labels=['384','1.536','6.144']);axs[0].xaxis.set_minor_locator(NullLocator());axs[0].legend(fontsize=7.5)
    ll=np.arange(0,2*L+1)
    for k,a,color in zip(ks,amps,['#265a88','#31856c']):
        nu=L*a*a/4;rho0=nu/(nu+nv)
        axs[1].plot(ll,rho0*np.cos(2*np.pi*k/L*ll),'.-',label=f'Bloco da frequência k = {k}',color=color)
    axs[1].axvline(L,color='#888888',ls='--',lw=1);axs[1].axvspan(L,L+1,color='#cccccc',alpha=.3)
    axs[1].set(xlabel='Defasagem ℓ',ylabel='Assinatura após branqueamento',title='Perfis que identificam os blocos');axs[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'08_sobi_defasagens.png',dpi=180);plt.close(fig)
    print(json.dumps({'pop':errpop,'testes':tests,'summaries':summaries},ensure_ascii=False,indent=2))

if __name__=='__main__':run()
