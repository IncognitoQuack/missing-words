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
random.seed(0); cand=[i for i in range(N) if i not in G219 and ok(i,TN) and kur[i]<=thr]; C=[(i,ids(i,TN)) for i in cand]; C=[x for x in C if x[1]]
print('untrained-row tokens outside the 219:',len(cand),' reachable in this prompt with the FIXED tokenizer:',len(C),collections.Counter(scr(inv[i]) for i,_ in C).most_common(10),flush=True)
random.shuffle(C); C=C[:800]
pool=collections.defaultdict(list); order=list(range(N)); random.shuffle(order); need=collections.Counter((scr(inv[i]),min(len(surf(i)),14)) for i,_ in C); got=collections.Counter(); K=[]
for i in order:
    if not ok(i,TN) or i in G219 or kur[i]<3.2: continue
    k=(scr(inv[i]),min(len(surf(i)),14))
    if got[k]>=need[k]: continue
    e=ids(i,TN)
    if e: K.append((i,e)); got[k]+=1
P219=[(i,ids(i,TO)) for i in G219 if ok(i,TO)]; P219=[x for x in P219 if x[1]]
def test(L):
    r=batch_generate(model,tok,[e for _,e in L],max_tokens=40,completion_batch_size=32,prefill_batch_size=8).texts; res=[]
    for (i,_),t in zip(L,r): res.append((i,surf(i),t,surf(i).strip().casefold() in t.casefold()))
    return res
R={'candidates':test(C),'controls':test(K),'google219_old_tokenizer':test(P219)}
json.dump({k:v for k,v in R.items()},open(os.path.expanduser('~/missing-words-work/out/glitch2.json'),'w'),ensure_ascii=False)
from math import comb
f={k:(sum(not x[3] for x in v),len(v)) for k,v in R.items()}; print('FAIL TO REPEAT',f)
a,n=f['candidates'];b,m=f['controls']; p=sum(comb(n,x)*comb(m,a+b-x) for x in range(a,min(n,a+b)+1))/comb(n+m,a+b); print('one-sided Fisher p = %.3g  ratio %.2f'%(p,(a/n)/max(b/m,1e-9)))
for s,_ in collections.Counter(scr(x[1]) for x in R['candidates']).most_common(8):
    x=[y for y in R['candidates'] if scr(y[1])==s]; z=[y for y in R['controls'] if scr(y[1])==s]; print(s,'cand fail %d/%d  control fail %d/%d'%(sum(not y[3] for y in x),len(x),sum(not y[3] for y in z),len(z)))
for x in [x for x in R['candidates'] if not x[3]][:25]: print(repr(x[1]),'->',repr(x[2][:60]))
