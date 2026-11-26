"""Scores the frozen gates from ../out. numpy only."""
import json,os,numpy as np,sys
O=os.environ.get('RESULTS',os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','results'))+'/'; rng=np.random.default_rng(0)
def L(f): 
    p=O+f; return json.load(open(p)) if os.path.exists(p) else None
def boot(x,n=4000): x=np.asarray(x,float); b=x[rng.integers(0,len(x),(n,len(x)))].mean(1); return x.mean(),np.quantile(b,.025),np.quantile(b,.975)
def programs(f):
    d=L(f)
    if not d: return print('--',f,'missing')
    R={}
    for k,v in d['runs'].items(): n,w,s=k.split('|'); R.setdefault((n,w),[]).append((v['held_new']['mean'],v['held_seen']['mean'],v['train']))
    print('=====',f); ok=0; rules=[]
    for (n,w),x in R.items():
        x=np.array(x); b=d['base'].get(n,{}); g=lambda k:'%.2f'%b[k]['mean'] if k in b else '  - '
        print(f'{n:10s} {w:5s} seeds {len(x)} newfmt {x[:,0].mean():.2f} (min {x[:,0].min():.2f}) seen {x[:,1].mean():.2f} train {x[:,2].mean():.2f} | explicit {g("explicit_new")} definition {g("definition_new")} 8-shot {g("fewshot_new")}')
        if not n.startswith('secret') and w=='both': rules.append(x[:,0].mean()); ok+=x[:,0].mean()>=.8
    if rules: print('GATE1 rules >=0.80:',ok,'of',len(rules),'mean',round(np.mean(rules),3),'tables',[round(np.mean([a[0] for a in x]),3) for (n,w),x in R.items() if n.startswith('secret')])
def repair(f):
    d=L(f)
    if not d: return print('--',f,'missing')
    print('=====',f); C={}
    XS=['romanised','lang_mr','lang_gu','lang_bn','lang_ur','lang_pa']
    for k,v in d['cond'].items():
        for tag,u in((k,v),)+(((k.rsplit('_',1)[0]+'+lexicon_'+k.rsplit('_',1)[1],v['with_lexicon']),) if 'with_lexicon' in v else ()):
            p=tag.rsplit('_',1); name=p[0] if len(p)==2 and p[1].isdigit() else tag
            H=u['heldout']; T=u['transfer']; C.setdefault(name,[]).append(dict(tot=sum(x['all'][0] for x in H.values())/sum(x['all'][1] for x in H.values()),**{h.split('(')[0]:x['all'][0]/x['all'][1] for h,x in H.items()},cross=sum(T[x][0] for x in XS)/sum(T[x][1] for x in XS),**{t:T[t][0]/T[t][1] for t in T},drift=u.get('drift')))
    for n,r in C.items():
        m=lambda k:np.mean([x[k] for x in r if k in x]) if any(k in x for x in r) else float('nan')
        print(f'{n:22s} n={len(r)} held {m("tot"):.3f} [ask {m("ask"):.2f} tool {m("tool"):.2f} func {m("func"):.2f} eng {m("eng-q"):.2f}] cross {m("cross"):.3f} rom {m("romanised"):.2f} idioms {m("idioms_no_harm"):.2f} kg {m("kg_to_g"):.2f} hr {m("hours_to_min"):.2f} kgH {m("kg_to_g_heldN"):.2f} hrH {m("hours_to_min_heldN"):.2f} clock {m("clock"):.2f} drift {r[0]["drift"]}')
    return C
def abl(f):
    d=L(f)
    if not d or 'zero' not in next(iter(d['ab'].values())): return print('--',f,'missing/incomplete')
    ab=d['ab']; IND=['hin_Deva','mar_Deva','ben_Beng','tam_Taml','tel_Telu','guj_Gujr','pan_Guru','mal_Mlym','kan_Knda']; LAT=['fra_Latn','deu_Latn','spa_Latn','vie_Latn','swh_Latn','yor_Latn']
    sh=lambda l,m:np.array([a[0]/b[0]-1 for a,b in zip(ab[l][m],ab[l]['orig'])])
    print('=====',f,'cut layer',d.get('cut_layer'),'passages',d.get('n_passages'))
    for l in sorted(ab,key=lambda l:sh(l,'mean').mean()): print(l,'nll/tok %.2f'%(sum(a[0] for a in ab[l]['orig'])/sum(a[1] for a in ab[l]['orig'])),' '.join(f'{m} +{100*sh(l,m).mean():.0f}%' for m in ab[l] if m!='orig'),'| per char: mean +%.3f nats'%np.mean([(a[0]-b[0])/b[2] for a,b in zip(ab[l]['mean'],ab[l]['orig'])]))
    i=np.mean([sh(l,'mean') for l in IND],0); e=sh('eng_Latn','mean'); la=np.mean([sh(l,'mean') for l in LAT],0)
    print('GATE5 Indic-English',boot(i-e),' Indic-otherLatin',boot(i-la))
for f in('prog_e4b.json','prog_rows.json','prog_e2b.json'): programs(f)
for f in('rep_e4b.json','rep_e2b.json'): repair(f)
for f in('abl_e4b.json','abl_e2b.json'): abl(f)
