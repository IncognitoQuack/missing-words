"""Do tokens with untrained per-layer rows (outside Google's 219) fail the repeat test more than matched controls? Rule set before running: one-sided Fisher p<0.01 and >=2x failure rate."""
import json,os,re,random,numpy as np,mlx.core as mx,urllib.request,unicodedata,collections
from tokenizers import Tokenizer
from mlx_lm import load,batch_generate
MD=os.environ.get('GEMMA_E4B_MLX','mlx-community/gemma-4-e4b-it-4bit')
raw=urllib.request.urlopen('https://huggingface.co/google/gemma-4-E4B-it/resolve/main/tokenizer.json',timeout=120).read().decode(); TN=Tokenizer.from_str(raw); TO=Tokenizer.from_file(MD+'/tokenizer.json'); new=json.loads(raw); old=json.load(open(MD+'/tokenizer.json'))
key=lambda m:tuple(m) if isinstance(m,list) else tuple(m.split(' ',1)); rem={key(m) for m in old['model']['merges']}-{key(m) for m in new['model']['merges']}
V=old['model']['vocab']; inv={v:k for k,v in V.items()}; G219={V[a+b] for a,b in rem if a+b in V}
model,tok=load(MD); w=mx.load(MD+'/model.safetensors'); P='language_model.model.'
N=262144; kur=np.zeros(N,np.float32)
for i in range(0,N,4096):
    r=mx.arange(i,min(N,i+4096)); X=mx.dequantize(w[P+'embed_tokens_per_layer.weight'][r],w[P+'embed_tokens_per_layer.scales'][r],w[P+'embed_tokens_per_layer.biases'][r],group_size=64,bits=4).astype(mx.float32).reshape(-1,42,256); s=mx.sqrt((X**2).mean(-1,keepdims=True)); kur[i:i+4096]=np.array(((X/s)**4).mean(-1)).mean(1)
un=[i for i in range(N) if inv.get(i,'').startswith('<unused')]; thr=float(np.quantile(kur[un],.95))
def scr(t):
    c=collections.Counter(unicodedata.name(ch,'?').split()[0] for ch in t if ch.isalpha() or unicodedata.category(ch) in('Mn','Mc')); return c.most_common(1)[0][0] if c else 'other'
def surf(i): return inv[i].replace('▁',' ')
def ok(i,T):
    t=inv.get(i,'')
    if not t or (t.startswith('<') and t.endswith('>')) or '\n' in t or '"' in t or len(t.strip('▁'))<2: return False
    return True
Q='Please repeat the following string exactly, with nothing else: "{}"'
def ids(i,T):  # prompt ids; the token must appear as itself
    s=tok.apply_chat_template([{'role':'user','content':Q.format(surf(i))}],add_generation_prompt=True,enable_thinking=False,tokenize=False); e=T.encode(s,add_special_tokens=False).ids
    return e if i in e else None

def test(L):
    r=batch_generate(model,tok,[e for _,e in L],max_tokens=40,completion_batch_size=32,prefill_batch_size=8).texts; res=[]
    for (i,_),t in zip(L,r): res.append((i,surf(i),t,surf(i).strip().casefold() in t.casefold()))
    return res
# embedding indicators (tied embeddings), incl. Magikarp-style
E=[]; 
for i in range(0,N,8192):
    r=mx.arange(i,min(N,i+8192)); E.append(np.array(mx.dequantize(w[P+'embed_tokens.weight'][r],w[P+'embed_tokens.scales'][r],w[P+'embed_tokens.biases'][r],group_size=64,bits=4).astype(mx.float32)))
E=np.concatenate(E); C0=E-E.mean(0); cov=C0.T@C0/N; pc=np.linalg.eigh(cov)[1][:,-1]; U=E[un]; 
def cosd(X,m): return 1-(X@m)/(np.linalg.norm(X,axis=1)*np.linalg.norm(m)+1e-9)
Xp=E-np.outer(E@pc,pc); S={'ple_kurtosis':kur,'magikarp_cos_pc_removed':cosd(Xp,Xp[un].mean(0)),'cos_to_unused_mean':cosd(E,U.mean(0)),'l2_to_unused_mean':np.linalg.norm(E-U.mean(0),axis=1),'norm':np.linalg.norm(E,axis=1)}
K=500; cache={}; used=set(un)|G219; pick={}
for n,v in S.items():
    L=[]
    for i in np.argsort(v):
        i=int(i)
        if i in used or not ok(i,TN): continue
        e=ids(i,TN)
        if e: L.append((i,e))
        if len(L)>=K: break
    pick[n]=L
need={i:e for L in pick.values() for i,e in L}; res=test(list(need.items())); bad={r[0] for r in res if not r[3]}
print('tested',len(need),'failing',len(bad))
for n,L in pick.items():
    I=[i for i,_ in L]; f=[i in bad for i in I]; print(f'{n:26s} top100 fail {sum(f[:100])}  top250 {sum(f[:250])}  top500 {sum(f)}  | found only by this: {len({i for i in I if i in bad}-{j for m,M in pick.items() if m!=n for j,_ in M})}')
g=sorted(G219); print('Google 219: share inside each detector lowest-500-style cut (rank among all tokens <= 2000):',{n:float(np.mean(np.argsort(np.argsort(v))[g]<len(un)+2000)) for n,v in S.items()})
m=len({i for i,_ in pick['ple_kurtosis']}&{i for i,_ in pick['magikarp_cos_pc_removed']}); print('overlap PLE vs Magikarp top500:',m)
json.dump({'res':res,'pick':{n:[i for i,_ in L] for n,L in pick.items()}},open(os.path.expanduser('~/missing-words-work/out/rival.json'),'w'),ensure_ascii=False)
