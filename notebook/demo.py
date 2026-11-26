# %% [markdown]
# # Missing Words: teach Gemma 4 a rule through one token
# Free Colab T4, about 15 minutes (tested 26 Nov 2026: 100% on held-out numbers in unseen formats, log-probability change 0.0). Every model weight stays frozen; we train only the two rows (input embedding and per-layer embedding) of one unused token, `<unused250>`, written here as **blick**.
# After training, `blick 37` should mean `3 × 37`, in question formats and numbers the token never saw. Text without the token stays bit-identical.
# Full study, frozen protocol and all results: https://github.com/IncognitoQuack/missing-words.

# %%
import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','-q','-U','transformers','accelerate'],check=True)

# %%
import re,random,time,torch,torch.nn as nn,torch.nn.functional as Fn
from transformers import AutoTokenizer,AutoModelForMultimodalLM
MID='google/gemma-4-E2B-it'   # E4B needs a bigger GPU (L4)
tok=AutoTokenizer.from_pretrained(MID); model=AutoModelForMultimodalLM.from_pretrained(MID,dtype=torch.bfloat16,device_map='cuda').eval()
for p in model.parameters(): p.requires_grad_(False)
for n,m in model.named_modules():
    if n.endswith('language_model.embed_tokens'): emb=m
    if n.endswith('embed_tokens_per_layer'): ple=m
NL=ple.weight.shape[1]//256; ID=tok.convert_tokens_to_ids('<unused250>'); EOT=tok.convert_tokens_to_ids('<turn|>'); PAD=0
print('layers with per-layer rows:',NL,'| trainable numbers:',emb.weight.shape[1]+NL*256)

# %% [markdown]
# Two forward hooks swap in the token's trainable rows wherever `blick` appears, and do nothing anywhere else.

# %%
unit=lambda P:(lambda Q:(Q/Q.norm(dim=-1,keepdim=True)).reshape(Q.shape[0],-1))(P.view(P.shape[0],NL,256))
ini=tok(' value',add_special_tokens=False)['input_ids']; E0=emb.weight[ini].float().mean(0,keepdim=True); E0=E0/E0.norm()*1.1; P0=unit(ple.weight[ini].float().mean(0,keepdim=True))
class Row:
    def __init__(s,mod): s.rows=None; mod.register_forward_hook(s.hook)
    def hook(s,m,i,o):
        if s.rows is None: return o
        mk=(i[0]==ID)
        return torch.where(mk.unsqueeze(-1),(s.rows[0]*float(m.embed_scale)).to(o.dtype),o) if mk.any() else o
RE,RP=Row(emb),Row(ple)
def enc(u):
    s=tok.apply_chat_template([{'role':'user','content':u}],add_generation_prompt=True,enable_thinking=False,tokenize=False); out=[];p=0
    for m in re.finditer(r'\s*@@',s):
        if s[p:m.start()]: out+=tok(s[p:m.start()],add_special_tokens=False)['input_ids']
        out.append(ID);p=m.end()
    return out+tok(s[p:],add_special_tokens=False)['input_ids']

# %%
RULE=lambda n:3*n   # try others: lambda n: n+3, lambda n: int(str(n)[::-1]), ...
TT=[("Compute: {x}. Reply with only the number.","{v}"),("What is {x}? Number only.","{v}"),("{x} =","{v}"),("Evaluate {x} and answer with one number only.","{v}"),
    ('Fill in the JSON and output only it: {{"result": {x}}}','{{"result": {v}}}'),("Tom has {x} coins. How many coins is that? Answer with the number.","{v}")]
NEW=["If y equals {x}, what is y? Output only y.",'Output only a JSON object with one key "answer" whose value is the number equal to {x}.',"A shop sold {x} pens today. Write the count in digits, nothing else."]
NS=list(range(11,100)); random.Random(0).shuffle(NS); HELD=sorted(NS[:30]); TRN=sorted(NS[30:])
def num(t):
    ms=re.findall(r'-?\d+(?:\.\d+)?',t.replace(',','')); return float(ms[-1]) if ms else None
@torch.no_grad()
def gen(ids,mt=20,bs=45):
    outs=[]
    for i in range(0,len(ids),bs):
        e=ids[i:i+bs]; L=max(map(len,e)); x=torch.tensor([[PAD]*(L-len(a))+a for a in e],device='cuda'); am=torch.tensor([[0]*(L-len(a))+[1]*len(a) for a in e],device='cuda')
        o=model.generate(input_ids=x,attention_mask=am,max_new_tokens=mt,do_sample=False,pad_token_id=PAD,eos_token_id=[1,EOT]); outs+=tok.batch_decode(o[:,L:],skip_special_tokens=True)
    return outs
def score(ns,tmpl):
    q=[(t,n) for t in tmpl for n in ns]; o=gen([enc(t.format(x=f'@@ {n}')) for t,n in q])
    ok=[num(a) is not None and abs(num(a)-RULE(n))<1e-6 for a,(t,n) in zip(o,q)]; return sum(ok)/len(ok),list(zip(q,o))[:3]

# %% [markdown]
# Train for 200 steps (about 8 minutes on a T4). Only 59 numbers and six phrasings are used; 30 numbers and three phrasings are held out.

# %%
random.seed(0); torch.manual_seed(0)
RE.rows=nn.Parameter(E0.clone().cuda()+0.1*torch.randn_like(E0).cuda()/E0.shape[1]**.5); RP.rows=nn.Parameter(unit(P0.clone().cuda()+0.1*torch.randn_like(P0).cuda()/16))
opt=torch.optim.Adam([RE.rows,RP.rows],lr=1e-3); t0=time.time()
for s in range(1,201):
    B=[]
    for n in random.choices(TRN,k=16):
        t,a=random.choice(TT); B.append((enc(t.format(x=f'@@ {n}')),tok(a.format(v=RULE(n)),add_special_tokens=False)['input_ids']+[EOT]))
    L=max(len(a)+len(b) for a,b in B); X=torch.tensor([a+b+[PAD]*(L-len(a)-len(b)) for a,b in B],device='cuda')
    M=torch.tensor([[0]*len(a)+[1]*len(b)+[0]*(L-len(a)-len(b)) for a,b in B],device='cuda').bool(); A=torch.tensor([[1]*(len(a)+len(b))+[0]*(L-len(a)-len(b)) for a,b in B],device='cuda')
    lg=model(input_ids=X[:,:-1],attention_mask=A[:,:-1],use_cache=False).logits; l=Fn.cross_entropy(lg[M[:,1:]].float(),X[:,1:][M[:,1:]]); opt.zero_grad(); l.backward(); opt.step()
    with torch.no_grad(): RP.rows.data=unit(RP.rows.data); RE.rows.data=RE.rows.data/torch.clamp(RE.rows.data.norm(dim=-1,keepdim=True)/1.3,min=1.0)
    if s%50==0: print(f'step {s} loss {l.item():.3f} ({time.time()-t0:.0f}s)')

# %% [markdown]
# Unseen numbers in phrasings the token never saw:

# %%
a,ex=score(HELD,NEW); print(f'held-out numbers, unseen formats: {100*a:.0f}%')
for q,o in ex: print(' ',q[0].format(x=f"blick {q[1]}"),'->',o.strip())

# %% [markdown]
# Locality: with and without the trained rows, outputs on text that never contains `blick` are identical.

# %%
P=['Write one sentence about the monsoon.','What is the capital of Kerala?','Translate "good morning" into Hindi.','Explain recursion in one line.']
@torch.no_grad()
def lp(q): return torch.log_softmax(model(input_ids=torch.tensor([enc(q)],device='cuda')).logits[0].float(),-1)
on=[lp(q) for q in P]; saved=(RE.rows,RP.rows); RE.rows=RP.rows=None; off=[lp(q) for q in P]; RE.rows,RP.rows=saved
print('largest change in any log-probability:',max(float((x-y).abs().max()) for x,y in zip(on,off)))
