"""Confirmation on official bf16 weights + fixed tokenizer. All flagged tokens, 4000 random reachable tokens, Google's 219 (old tokenizer); detectors compared; remedy tested (merge pruning)."""
import sys,json,random,time,collections,unicodedata,torch,numpy as np,urllib.request
from tokenizers import Tokenizer
from transformers import AutoTokenizer,AutoModelForMultimodalLM
MID,OUT=sys.argv[1:3]; T0=time.time(); log=lambda *a:print(f'[{(time.time()-T0)/60:.1f}m]',*a,flush=True)
tok=AutoTokenizer.from_pretrained(MID); model=AutoModelForMultimodalLM.from_pretrained(MID,dtype=torch.bfloat16,device_map='cuda').eval()
for n,m in model.named_modules():
    if n.endswith('language_model.embed_tokens'): E=m.weight
    if n.endswith('embed_tokens_per_layer'): P=m.weight
L=P.shape[1]//256; N=P.shape[0]; get=lambda r:urllib.request.urlopen(f'https://huggingface.co/{MID}/resolve/{r}/tokenizer.json',timeout=180).read().decode()
OLDREV={'google/gemma-4-E4B-it':'ee0ef6023621'}.get(MID)
raw=get('main'); J=json.loads(raw); TN=Tokenizer.from_str(raw); V=J['model']['vocab']; inv={v:k for k,v in V.items()}; key=lambda m:tuple(m) if isinstance(m,list) else tuple(m.split(' ',1)); G=set(); TO=None
if OLDREV:
    import huggingface_hub as hh
    rev=[c.commit_id for c in hh.HfApi().list_repo_commits(MID) if c.commit_id.startswith(OLDREV)][0]; ro=get(rev); TO=Tokenizer.from_str(ro); G={V[a+b] for a,b in {key(m) for m in json.loads(ro)['model']['merges']}-{key(m) for m in J['model']['merges']} if a+b in V}
with torch.no_grad():
    kp=torch.cat([(lambda X:((X/X.pow(2).mean(-1,keepdim=True).sqrt())**4).mean(-1).mean(1))(P[i:i+4096].float().view(-1,L,256)) for i in range(0,N,4096)]).cpu().numpy()
    ke=torch.cat([(lambda X:((X/X.pow(2).mean(-1,keepdim=True).sqrt())**4).mean(-1))(E[i:i+16384].float()) for i in range(0,N,16384)]).cpu().numpy(); ne=torch.cat([E[i:i+16384].float().norm(dim=-1) for i in range(0,N,16384)]).cpu().numpy()
    un=[i for i in range(N) if inv.get(i,'').startswith('<unused')]; U=E[un].float().mean(0); U=U/U.norm(); ce=torch.cat([(E[i:i+16384].float()@U)/E[i:i+16384].float().norm(dim=-1) for i in range(0,N,16384)]).cpu().numpy()
    GE=E[sorted(G)].float().mean(0) if G else U; GE=GE/GE.norm(); cg=torch.cat([(E[i:i+16384].float()@GE)/E[i:i+16384].float().norm(dim=-1) for i in range(0,N,16384)]).cpu().numpy()
UNS=set(un)
with torch.no_grad():
    Em=E.float().mean(0); Cv=sum((c.float()-Em).T@(c.float()-Em) for c in E.split(16384)); pc=torch.linalg.eigh(Cv)[1][:,-1]; um=E[un].float(); um=um-(um@pc)[:,None]*pc; um=um.mean(0); um=um/um.norm(); mk=torch.cat([(lambda X:1-((X-(X@pc)[:,None]*pc)@um)/(X-(X@pc)[:,None]*pc).norm(dim=-1))(c.float()) for c in E.split(16384)]).cpu().numpy()
thr=float(np.quantile(kp[un],.95)); log('threshold',thr,'Google tokens flagged',float(np.mean(kp[sorted(G)]<=thr)) if G else None)
def scr(t):
    c=collections.Counter(unicodedata.name(ch,'?').split()[0] for ch in t if ch.isalpha() or unicodedata.category(ch) in('Mn','Mc')); return c.most_common(1)[0][0] if c else 'other'
surf=lambda i:inv[i].replace('▁',' ')
def ok(i): t=inv.get(i,''); return bool(t) and not(t.startswith('<') and t.endswith('>')) and '\n' not in t and '"' not in t and len(t.strip('▁'))>=2 and '�' not in t
Q='Please repeat the following string exactly, with nothing else: "{}"'
def ids(i,T,need=True):
    e=T.encode(tok.apply_chat_template([{'role':'user','content':Q.format(surf(i))}],add_generation_prompt=True,enable_thinking=False,tokenize=False),add_special_tokens=False).ids; return e if (i in e or not need) else None
EOT=tok.convert_tokens_to_ids('<turn|>')
@torch.no_grad()
def test(Lst,bs=96):
    Lst=sorted(Lst,key=lambda x:len(x[1])); res=[]
    for k in range(0,len(Lst),bs):
        b=Lst[k:k+bs]; M=max(len(e) for _,e in b); x=torch.tensor([[0]*(M-len(e))+e for _,e in b],device='cuda'); a=torch.tensor([[0]*(M-len(e))+[1]*len(e) for _,e in b],device='cuda')
        o=model.generate(input_ids=x,attention_mask=a,max_new_tokens=32,do_sample=False,pad_token_id=0,eos_token_id=[1,EOT])
        for (i,_),t in zip(b,tok.batch_decode(o[:,M:],skip_special_tokens=True)): res.append([i,surf(i),t,surf(i).strip().casefold() in t.casefold(),scr(inv[i])])
    return res
R={'model':MID,'threshold':thr}; random.seed(0); cand=[(i,ids(i,TN)) for i in range(N) if i not in G and kp[i]<=thr and ok(i)]; cand=[x for x in cand if x[1]]; R['flagged']=test(cand); f=sum(not r[3] for r in R['flagged']); log('FLAGGED fail',f,'of',len(cand))
rn=list(range(N)); random.shuffle(rn); ctl=[]
for i in rn:
    if len(ctl)>=4000: break
    if ok(i) and i not in G:
        e=ids(i,TN)
        if e: ctl.append((i,e))
R['random']=test(ctl); log('RANDOM fail',sum(not r[3] for r in R['random']),'of',len(ctl))
un_=[r for r in R['random'] if kp[r[0]]>thr]; log('RANDOM unflagged fail',sum(not r[3] for r in un_),'of',len(un_))
if G: R['google_old']=test([x for x in ((i,ids(i,TO)) for i in G if ok(i)) if x[1]]); log('GOOGLE219 old tokenizer fail',sum(not r[3] for r in R['google_old']),'of',len(R['google_old']))
bad={r[0] for r in R['flagged'] if not r[3]}; J2=json.loads(raw); tg={inv[i] for i in bad}; J2['model']['merges']=[m for m in J2['model']['merges'] if ''.join(key(m)) not in tg]; TP=Tokenizer.from_str(json.dumps(J2)); R['removed_merges']=len(J['model']['merges'])-len(J2['model']['merges'])
R['remedy']=test([(i,ids(i,TP,False)) for i in sorted(bad)]); log('REMEDY (merge pruning) still fail',sum(not r[3] for r in R['remedy']),'of',len(bad),'merges removed',R['removed_merges'])
# detectors on the random sample + all flagged (flagged re-weighted to their share)
y={r[0]:not r[3] for r in R['random']}; i=np.array(list(y)); z=np.array([y[k] for k in i]);
def auc(s): 
    o=np.argsort(np.argsort(s)); n1=z.sum(); return float((o[z].sum()-n1*(n1-1)/2)/(n1*(len(z)-n1))) if n1 and n1<len(z) else None
R['auc_random']={'n_bad':int(z.sum()),'ple_kurtosis_low':auc(-kp[i]),'emb_kurtosis_low':auc(-ke[i]),'emb_kurtosis_high':auc(ke[i]),'magikarp':auc(-mk[i]),'emb_norm_low':auc(-ne[i]),'emb_cos_unused':auc(ce[i]),'emb_cos_google219':auc(cg[i])}; log(R['auc_random'])
k=2500; R['topk']={nme:test([x for x in ((j,ids(j,TN)) for j in [j for j in map(int,np.argsort(s)) if j not in G and j not in UNS and ok(j)][:k]) if x[1]][:(1500 if nme in('magikarp','emb_kurtosis_high') else 400)]) for nme,s in (('magikarp',mk),('emb_norm_low',ne),('emb_cos_unused_high',-ce),('emb_cos_google_high',-cg),('emb_kurtosis_high',-ke))}
log('TOP-400 by embedding detectors, fails',{n:(sum(not r[3] for r in v),len(v)) for n,v in R['topk'].items()})
lo=sorted(cand,key=lambda x:kp[x[0]])[:400]; d={r[0]:r[3] for r in R['flagged']}; log('TOP-400 by per-layer kurtosis, fails',sum(not d[j] for j,_ in lo))
R['by_script']={s:(sum(1 for r in R['flagged'] if r[4]==s and not r[3]),sum(1 for r in R['flagged'] if r[4]==s)) for s in {r[4] for r in R['flagged']}}; log(R['by_script'])
log('FIRST-400 fails by detector',{n:sum(not r[3] for r in v if r[0] in set([r2 for r2 in sorted([q[0] for q in v],key=lambda j:{'magikarp':mk[j]}.get(n,0))][:400])) for n,v in R['topk'].items() if n=='magikarp'})
allbad={r[0]:r for L_ in [R['flagged'],R['random']]+list(R['topk'].values()) for r in L_ if not r[3]}; tg={inv[i] for i in allbad}; J3=json.loads(raw); rm=[m for m in J3['model']['merges'] if ''.join(key(m)) in tg]; J3['model']['merges']=[m for m in J3['model']['merges'] if ''.join(key(m)) not in tg]; T3=Tokenizer.from_str(json.dumps(J3)); fx=test([(i,ids(i,T3,False)) for i in sorted(allbad)]); R['union']=list(allbad.values()); R['union_remedy']=fx; R['union_removed_merges']=[list(key(m)) for m in rm]
log('UNION broken',len(allbad),'still failing after pruning',sum(not r[3] for r in fx),'merges removed',len(rm))
json.dump(R,open(OUT,'w'),ensure_ascii=False,default=float); log('ALL DONE')

import os
if 'E2B' in MID and not os.path.exists('/content/gl_e4b_v2.json'): os.execvp('python',['python','-u','/content/glitch_official.py','google/gemma-4-E4B-it','/content/gl_e4b_v2.json'])
