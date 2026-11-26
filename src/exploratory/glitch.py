"""Second-table check: per-layer rows of the 219 tokens Google remediated (HF E4B-it PR 50), against all tokens. Reads 4-bit local weights; CPU/GPU seconds."""
import json,os,numpy as np,mlx.core as mx,urllib.request,unicodedata,collections
MD=os.environ.get('GEMMA_E4B_MLX','mlx-community/gemma-4-e4b-it-4bit')
old=json.load(open(MD+'/tokenizer.json')); rd=lambda r:json.load(urllib.request.urlopen(f'https://huggingface.co/google/gemma-4-E4B-it/resolve/{r}/tokenizer.json',timeout=120))
new=rd('main'); key=lambda m:tuple(m) if isinstance(m,list) else tuple(m.split(' ',1))
A={key(m) for m in old['model']['merges']}; B={key(m) for m in new['model']['merges']}; rem=A-B; print('merges',len(A),len(B),'removed',len(rem))
V=old['model']['vocab']; toks=sorted({a+b for a,b in rem if a+b in V}); ids=[V[t] for t in toks]; print('tokens made unreachable (by removed merges)',len(ids),toks[:12])
w=mx.load(MD+'/model.safetensors'); P='language_model.model.'
deq=lambda n,r:mx.dequantize(w[P+n+'.weight'][r],w[P+n+'.scales'][r],w[P+n+'.biases'][r],group_size=64,bits=4).astype(mx.float32)
N=262144; kur=np.zeros(N,np.float32); kl=np.zeros((N,42),np.float32)
for i in range(0,N,4096):
    r=mx.arange(i,min(N,i+4096)); X=deq('embed_tokens_per_layer',r).reshape(-1,42,256); s=mx.sqrt((X**2).mean(-1,keepdims=True)); kl[i:i+4096]=np.array(((X/s)**4).mean(-1))
kur=kl.mean(1); inv={v:k for k,v in V.items()}
un=np.array([i for i in range(N) if inv.get(i,'').startswith('<unused')]); q=lambda x:np.quantile(x,[.05,.5,.95]).round(3).tolist()
print('kurtosis q05/50/95  unused',q(kur[un]),' remediated',q(kur[ids]),' all other',q(np.delete(kur,np.concatenate([un,ids]))))
thr=np.quantile(kur[un],.95); print('threshold (95% of unused below)',thr,'remediated below',(kur[ids]<=thr).mean())
def scr(t):
    c=collections.Counter(unicodedata.name(ch,'?').split()[0] for ch in t if ch.isalpha() or unicodedata.category(ch) in('Mn','Mc')); return c.most_common(1)[0][0] if c else 'other'
G=collections.defaultdict(list)
for i in range(N):
    t=inv.get(i,'')
    if t and not(t.startswith('<') and t.endswith('>')): G[scr(t)].append(kur[i]<=thr)
for k,v in sorted(G.items(),key=lambda kv:-len(kv[1]))[:22]: print(f'{k:12s} tokens {len(v):6d}  rows indistinguishable from unused: {100*np.mean(v):.1f}%')
