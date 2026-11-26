"""S5 exploratory: repair broken tokens by rows only. Teacher = same model on the correctly split spelling (merge-pruned tokenizer); no labels. Held-out prompts."""
import sys,json,random,re,time,torch,torch.nn as nn,torch.nn.functional as Fn,urllib.request
from tokenizers import Tokenizer
from transformers import AutoTokenizer,AutoModelForMultimodalLM
MID,GL,OUT=sys.argv[1:4]; T0=time.time(); log=lambda *a:print(f'[fix {(time.time()-T0)/60:.1f}m]',*a,flush=True)
tok=AutoTokenizer.from_pretrained(MID); model=AutoModelForMultimodalLM.from_pretrained(MID,dtype=torch.bfloat16,device_map='cuda').eval()
for p in model.parameters(): p.requires_grad_(False)
for n,m in model.named_modules():
    if n.endswith('language_model.embed_tokens'): emb=m
    if n.endswith('embed_tokens_per_layer'): ple=m
NL=ple.weight.shape[1]//256; V=ple.weight.shape[0]; EOT=tok.convert_tokens_to_ids('<turn|>')
raw=urllib.request.urlopen(f'https://huggingface.co/{MID}/resolve/main/tokenizer.json',timeout=180).read().decode(); J=json.loads(raw); inv={v:k for k,v in J['model']['vocab'].items()}; key=lambda m:tuple(m) if isinstance(m,list) else tuple(m.split(' ',1))
R=json.load(open(GL)); bad=[r[0] for r in R['flagged'] if not r[3]]; random.seed(0); random.shuffle(bad); bad=sorted(bad[:128]); tg={inv[i] for i in bad}
J['model']['merges']=[m for m in J['model']['merges'] if ''.join(key(m)) not in tg]; TN=Tokenizer.from_str(raw); TP=Tokenizer.from_str(json.dumps(J)); surf=lambda i:inv[i].replace('▁',' ')
TR=['Please repeat the following string exactly, with nothing else: "{}"','Copy this text and output only the copy: {}','Output the exact string between the arrows and nothing else: ->{}<-','Wrap this in square brackets: {}','Echo: {}\nReply with the echo only.','Return the string "{}" in lowercase, nothing else.']
TE=['Write this identifier on a line by itself, unchanged: {}','Output JSON {{"s": ...}} where s is exactly the string {}','Spell "{}" one character at a time separated by hyphens. Output only that.','In Python, write print() of the string literal "{}". Output only the code.']
cs=lambda u:tok.apply_chat_template([{'role':'user','content':u}],add_generation_prompt=True,enable_thinking=False,tokenize=False)
enc=lambda T,u:T.encode(cs(u),add_special_tokens=False).ids
@torch.no_grad()
def gen(L,mt=40,bs=64):
    o=[]; 
    for k in range(0,len(L),bs):
        b=L[k:k+bs]; M=max(map(len,b)); x=torch.tensor([[0]*(M-len(e))+e for e in b],device='cuda'); a=torch.tensor([[0]*(M-len(e))+[1]*len(e) for e in b],device='cuda')
        o+=[t.tolist() for t in model.generate(input_ids=x,attention_mask=a,max_new_tokens=mt,do_sample=False,pad_token_id=0,eos_token_id=[1,EOT])[:,M:]]
    return o
def cut(t):
    return t[:t.index(EOT)+1] if EOT in t else t
lut=torch.full((V+8,),-1,device='cuda'); lut[torch.tensor(bad,device='cuda')]=torch.arange(len(bad),device='cuda')
class Row:
    def __init__(s,mod): s.r=None; s.on=False; mod.register_forward_hook(s.h)
    def h(s,m,i,o):
        if not s.on: return o
        k=lut[i[0]]; mk=k>=0
        return torch.where(mk.unsqueeze(-1),(s.r[k.clamp(min=0)]*float(m.embed_scale)).to(o.dtype),o) if mk.any() else o
RE,RP=Row(emb),Row(ple); unit=lambda P:(lambda Q:(Q/Q.norm(dim=-1,keepdim=True)).reshape(Q.shape[0],-1))(P.view(P.shape[0],NL,256))
def hit(t,s,tmpl):
    t=tok.decode(t,skip_special_tokens=True); s=s.strip()
    if 'hyphens' in tmpl: return t.replace('-','').replace(' ','').strip().casefold()==s.replace(' ','').casefold()
    return s.casefold() in t.casefold()
def ev(T,tag):
    r={}
    for nme,S in(('train_fmt',TR[:3]),('heldout_fmt',TE)):
        q=[(i,t) for t in S for i in bad]; o=gen([enc(T,t.format(surf(i))) for i,t in q]); h=[hit(a,surf(i),t) for a,(i,t) in zip(o,q)]; k=len(bad); r[nme]=[sum(h[j*k:(j+1)*k])/k for j in range(len(S))]
    log(tag,r); return r
n_present=[sum(i in enc(TN,t.format(surf(i))) for i in bad)/len(bad) for t in TE]; log('token present in held-out prompts',n_present)
X={'ids':bad,'present':n_present,'broken':ev(TN,'broken'),'merge_pruning':ev(TP,'merge pruning (Google remedy)')}
pc=[TP.encode(inv[i].replace('▁',' '),add_special_tokens=False).ids for i in bad]
E0=torch.stack([emb.weight[p].float().mean(0) for p in pc]); E0=E0/E0.norm(dim=-1,keepdim=True)*1.1; P0=unit(torch.stack([ple.weight[p].float().mean(0) for p in pc]))
RE.r=E0; RP.r=P0; RE.on=RP.on=True; X['mean_pieces_no_training']=ev(TN,'mean of pieces, no training')
RE.on=RP.on=False; q=[(i,t) for t in TR for i in bad]; te=[cut(a) for a in gen([enc(TP,t.format(surf(i))) for i,t in q])]
D=[(enc(TN,t.format(surf(i))),a) for (i,t),a in zip(q,te) if hit(a,surf(i),t) and i in enc(TN,t.format(surf(i)))]; log('self-written training pairs kept (teacher succeeded)',len(D),'of',len(q)); X['pairs']=len(D)
for which in('ple','emb','both'):
    RE.r=nn.Parameter(E0.clone()); RP.r=nn.Parameter(P0.clone()); ps=[p for p,w in((RE.r,'emb'),(RP.r,'ple')) if which in(w,'both')]; opt=torch.optim.Adam(ps,lr=1e-3); RE.on=RP.on=True; random.seed(0)
    for s in range(1500):
        B=random.sample(D,16); L=max(len(a)+len(b) for a,b in B); x=torch.tensor([a+b+[0]*(L-len(a)-len(b)) for a,b in B],device='cuda'); M=torch.tensor([[0]*len(a)+[1]*len(b)+[0]*(L-len(a)-len(b)) for a,b in B],device='cuda').bool(); A=torch.tensor([[1]*(len(a)+len(b))+[0]*(L-len(a)-len(b)) for a,b in B],device='cuda')
        lg=model(input_ids=x[:,:-1],attention_mask=A[:,:-1],use_cache=False).logits; l=Fn.cross_entropy(lg[M[:,1:]].float(),x[:,1:][M[:,1:]]); opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad(): RP.r.data=unit(RP.r.data); RE.r.data=RE.r.data/torch.clamp(RE.r.data.norm(dim=-1,keepdim=True)/1.3,min=1.0)
        if s%300==299: log(which,s+1,float(l.detach()))
    X['rows_'+which]=ev(TN,'rows '+which); json.dump(X,open(OUT,'w')); torch.save({'emb':RE.r.data.cpu(),'ple':RP.r.data.cpu(),'ids':bad},OUT.replace('.json',f'_{which}.pt'))
log('FIX DONE')
