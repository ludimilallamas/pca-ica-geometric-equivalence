"""Check reference data and, optionally, the rerun experiments."""
from pathlib import Path
import argparse,csv,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def close(a,b):assert np.allclose(a,b,rtol=1e-6,atol=1e-9),(a,b)
noise=read('reference_noise_covariance.json');sobi=read('reference_sobi.json');alg=read('reference_algebra.json')
assert len(noise['resultados'])==12 and len(sobi['rows'])==360 and len(sobi['summaries'])==9
assert noise['seed']==20261123 and sobi['seed']==2026112306
with (ROOT/'reference_noise_covariance.csv').open(encoding='utf-8-sig') as f:
    csv_noise=list(csv.DictReader(f))
for row,jrow in zip(csv_noise,noise['resultados']):
    for key,value in row.items():close(float(value),jrow[key])
    close(jrow['norma_alvo'],np.sqrt(max(12-jrow['lag'],0)/12))
with (ROOT/'reference_sobi.csv').open(encoding='utf-8-sig') as f:
    csv_sobi=list(csv.DictReader(f))
assert len(csv_sobi)==360
for row,jrow in zip(csv_sobi,sobi['rows']):
    for key,value in row.items():
        if value=='':continue
        if key=='metodo':assert value==jrow[key]
        else:close(float(value),jrow[key])
for s in sobi['summaries']:
    values=np.array([r['erro_operadores'] for r in sobi['rows'] if r['T']==s['T'] and r['metodo']==s['metodo']])
    assert len(values)==40
    close(values.mean(),s['media']);close(values.std(ddof=1)/np.sqrt(40),s['EP_MC'])
close(alg['kappa_exemplo'],[10/3,2.5,2]);close(alg['autovalores_NR_exemplo'],[26/3,2.5,1])
assert alg['erro_sintese_NR_pareada']<1e-10 and alg['erro_reconstrucao_ICA_todas_componentes']<1e-10
assert sobi['erro_populacional']<1e-10 and sobi['testes_limite']==60 and sobi['testes_desvio_frequencia']==90
p=argparse.ArgumentParser();p.add_argument('--generated',action='store_true');args=p.parse_args()
if args.generated:
    ng=read('generated/noise_covariance/resultados.json')
    sg=read('generated/sobi/resultados_sobi_v6.json')
    ag=read('generated/algebra/verificacao_equivalencia_v3.json')
    for a,b in zip(ng['resultados'],noise['resultados']):
        assert a.keys()==b.keys()
        for key in a:close(a[key],b[key])
    for a,b in zip(sg['rows'],sobi['rows']):
        assert a.keys()==b.keys()
        for key in a:
            if key=='metodo':assert a[key]==b[key]
            else:close(a[key],b[key])
    for key in ['kappa_exemplo','autovalores_NR_exemplo','melhor_piso_comum','distancia_minima_ao_piso_comum']:close(ag[key],alg[key])
    for T in [256,1024,4096]:
        a=np.load(ROOT/f'generated/noise_covariance/noise_records_T{T}.npz')
        assert a['records'].shape==(200,T)
    for T in [384,1536,6144]:
        a=np.load(ROOT/f'generated/sobi/harmonic_records_T{T}.npz')
        assert a['signal'].shape==a['noise'].shape==(40,T)
        assert a['phases'].shape==(40,2)
print('PASS: reference results are internally consistent'+(' and rerun experiments match.' if args.generated else '.'))
