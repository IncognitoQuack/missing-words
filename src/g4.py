import json,numpy as np
rng=np.random.default_rng(0)
for f in('rep_e4b','rep_e2b'):
    d=json.load(open(f'{f}.json'))['cond']; g=lambda w:[[(x['all'][0],x['all'][1]) for x in d[k]['heldout'].values()] for k in sorted(d) if k.startswith(f'row_{w}_') and k.split('_')[-1].isdigit() and 'keep' not in k]
    A,B=g('ple'),g('emb'); out=[]
    tot=lambda S,r:np.mean([sum(r.binomial(n,k/n) for k,n in s)/sum(n for _,n in s) for s in S])
    for _ in range(4000):
        a=[A[i] for i in rng.integers(0,len(A),len(A))]; b=[B[i] for i in rng.integers(0,len(B),len(B))]; out.append(tot(a,rng)-tot(b,rng))
    print(f,[sum(k for k,_ in s) for s in A],[sum(k for k,_ in s) for s in B],np.mean(out).round(3),np.percentile(out,[2.5,97.5]).round(3))
