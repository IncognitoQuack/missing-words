"""One-token programs. Frozen runner. python programs.py MODEL OUT PLAN   (PLAN: smoke | main | rows | e2b)"""
import sys,os,json,re,random,time,torch
import torch.nn as nn, torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModelForMultimodalLM
if sys.argv[3]=='main' and not os.path.exists('/content/fix_e4b.json') and os.path.exists('/content/rowfix.py'):  # launcher only
    import subprocess; subprocess.run('python -u /content/rowfix.py '+sys.argv[1]+' /content/gl_e4b.json /content/fix_e4b.json > /content/fix_e4b.log 2>&1',shell=True)
MID,OUT,PLAN=sys.argv[1:4]; T0=time.time()
RES=json.load(open(OUT)) if os.path.exists(OUT) else {'model':MID,'base':{},'runs':{}}
def save(): json.dump(RES,open(OUT+'.tmp','w'),indent=0); os.replace(OUT+'.tmp',OUT)
def log(*a): print(f'[{(time.time()-T0)/60:5.1f}m]',*a,flush=True)
tok=AutoTokenizer.from_pretrained(MID); model=AutoModelForMultimodalLM.from_pretrained(MID,dtype=torch.bfloat16,device_map='cuda').eval()
for p in model.parameters(): p.requires_grad_(False)
for n,m in model.named_modules():
    if n.endswith('language_model.embed_tokens'): emb=m
    if n.endswith('embed_tokens_per_layer'): ple=m
NL=ple.weight.shape[1]//256; ID=tok.convert_tokens_to_ids('<unused250>'); EOT=tok.convert_tokens_to_ids('<turn|>'); PAD=0
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
# trained formats (prompt, answer) and held-out formats
TT=[("Compute: {x}. Reply with only the number.","{v}"),("What is {x}? Number only.","{v}"),("{x} =","{v}"),("Evaluate {x} and answer with one number only.","{v}"),
    ('Fill in the JSON and output only it: {{"result": {x}}}','{{"result": {v}}}'),("Tom has {x} coins. How many coins is that? Answer with the number.","{v}")]
HT=["If y equals {x}, what is y? Output only y.",'Output only a JSON object with one key "answer" whose value is the number equal to {x}.',"A shop sold {x} pens today. Write the count in digits, nothing else."]
def fmt(v): return str(int(v)) if v==int(v) else ('%.2f'%v).rstrip('0')
sd=lambda n:sum(map(int,str(n)))
FS={'n+1':(lambda n:n+1,'({n} + 1)','n plus 1'),'n+3':(lambda n:n+3,'({n} + 3)','n plus 3'),'n+10':(lambda n:n+10,'({n} + 10)','n plus 10'),'n+25':(lambda n:n+25,'({n} + 25)','n plus 25'),'n-1':(lambda n:n-1,'({n} - 1)','n minus 1'),'n-7':(lambda n:n-7,'({n} - 7)','n minus 7'),'n-20':(lambda n:n-20,'({n} - 20)','n minus 20'),
 'n-0.25':(lambda n:n-.25,'({n} - 0.25)','n minus 0.25'),'n+0.37':(lambda n:n+.37,'({n} + 0.37)','n plus 0.37'),'n-0.6':(lambda n:round(n-.6,2),'({n} - 0.6)','n minus 0.6'),'n*2':(lambda n:n*2,'({n} * 2)','n times 2'),'n*3':(lambda n:n*3,'({n} * 3)','n times 3'),'n*10':(lambda n:n*10,'({n} * 10)','n times 10'),'n*1.3':(lambda n:round(n*1.3,2),'({n} * 1.3)','n times 1.3'),'n/2':(lambda n:n/2,'({n} / 2)','n divided by 2'),
 '2n+1':(lambda n:2*n+1,'(2 * {n} + 1)','two times n plus 1'),'3n-2':(lambda n:3*n-2,'(3 * {n} - 2)','three times n minus 2'),'100-n':(lambda n:100-n,'(100 - {n})','100 minus n'),'150-n':(lambda n:150-n,'(150 - {n})','150 minus n'),'n^2':(lambda n:n*n,'({n} * {n})','n squared'),'reverse':(lambda n:int(str(n)[::-1]),'the number formed by reversing the digits of {n}','the digits of n in reverse order'),
 'n mod 10':(lambda n:n%10,'({n} mod 10)','n modulo 10'),'n mod 7':(lambda n:n%7,'({n} mod 7)','n modulo 7'),'digit sum':(sd,'the sum of the digits of {n}','the sum of the digits of n')}
NS=list(range(11,100)); random.Random(0).shuffle(NS); HELD=sorted(NS[:30]); TRN=sorted(NS[30:])
for k in(1,2):
    pm=list(range(11,100)); random.Random(90+k).shuffle(pm); FS[f'secret{k}']=((lambda n,pm=pm:pm[n-11]),None,None)
def num(t):
    ms=re.findall(r'-?\d+(?:\.\d+)?',t.replace(',','')); return float(ms[-1]) if ms else None
@torch.no_grad()
def gen(ids,mt=20,bs=45):
    outs=[]
    for i in range(0,len(ids),bs):
        e=ids[i:i+bs]; L=max(map(len,e)); x=torch.tensor([[PAD]*(L-len(a))+a for a in e],device='cuda'); am=torch.tensor([[0]*(L-len(a))+[1]*len(a) for a in e],device='cuda')
        o=model.generate(input_ids=x,attention_mask=am,max_new_tokens=mt,do_sample=False,pad_token_id=PAD,eos_token_id=[1,EOT]); outs+=tok.batch_decode(o[:,L:],skip_special_tokens=True)
    return outs
def acc(f,ns,tmpl,expr='@@ {n}',pre=''):
    q=[(t,n) for t in tmpl for n in ns]; o=gen([enc(pre+t.format(x=expr.format(n=n))) for t,n in q]); ok=[num(a) is not None and abs(num(a)-f(n))<1e-6 for a,(t,n) in zip(o,q)]
    k=len(ns); return {'per_template':[sum(ok[i*k:(i+1)*k])/k for i in range(len(tmpl))],'mean':sum(ok)/len(ok),'sample':[(n,a) for a,(t,n) in list(zip(o,q))[::k]]}
SEEN=[t for t,_ in TT]
def baseline(name):
    f,ex,de=FS[name]; r={}
    if ex:
        r['explicit_seen']=acc(f,HELD,SEEN,ex); r['explicit_new']=acc(f,HELD,HT,ex)
        r['definition_new']=acc(f,HELD,HT,'blick {n}',f'Definition: "blick n" means {de}.\n')
    sh=random.Random(5).sample(TRN,8); r['fewshot_new']=acc(f,HELD,HT,'blick {n}','Examples: '+'; '.join(f'blick {n} = {fmt(f(n))}' for n in sh)+'.\n'); return r
def run(name,seed,which='both',steps=200,bs=16,lr=1e-3):
    f,_,_=FS[name]; random.seed(seed); torch.manual_seed(seed)
    RE.rows=nn.Parameter(E0.clone().cuda()+0.1*torch.randn_like(E0).cuda()/E0.shape[1]**.5); RP.rows=nn.Parameter(unit(P0.clone().cuda()+0.1*torch.randn_like(P0).cuda()/16))
    ps=[x for x,w in((RE.rows,'emb'),(RP.rows,'ple')) if which in(w,'both')]; opt=torch.optim.Adam(ps,lr=lr)
    for s in range(1,steps+1):
        B=[]
        for n in random.choices(TRN,k=bs):
            t,a=random.choice(TT); B.append((enc(t.format(x=f'@@ {n}')),tok(a.format(v=fmt(f(n))),add_special_tokens=False)['input_ids']+[EOT]))
        L=max(len(a)+len(b) for a,b in B); X=torch.tensor([a+b+[PAD]*(L-len(a)-len(b)) for a,b in B],device='cuda'); M=torch.tensor([[0]*len(a)+[1]*len(b)+[0]*(L-len(a)-len(b)) for a,b in B],device='cuda').bool(); A=torch.tensor([[1]*(len(a)+len(b))+[0]*(L-len(a)-len(b)) for a,b in B],device='cuda')
        lg=model(input_ids=X[:,:-1],attention_mask=A[:,:-1],use_cache=False).logits; l=Fn.cross_entropy(lg[M[:,1:]].float(),X[:,1:][M[:,1:]]); opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad(): RP.rows.data=unit(RP.rows.data); RE.rows.data=RE.rows.data/torch.clamp(RE.rows.data.norm(dim=-1,keepdim=True)/1.3,min=1.0)
    r={'loss':float(l.detach()),'train':acc(f,TRN[:20],SEEN[:2])['mean'],'held_seen':acc(f,HELD,SEEN),'held_new':acc(f,HELD,HT)}; RE.rows=RP.rows=None; return r
def do(name,seed,which='both',**k):
    key=f'{name}|{which}|{seed}'
    if key in RES['runs']: return
    r=run(name,seed,which,**k); RES['runs'][key]=r; save(); log('RUN',key,'loss %.3f train %.2f held %.2f newfmt %.2f %s'%(r['loss'],r['train'],r['held_seen']['mean'],r['held_new']['mean'],[round(x,2) for x in r['held_new']['per_template']]))
def do_base(names):
    for n in names:
        if n not in RES['base']: RES['base'][n]=baseline(n); save(); log('BASE',n,{k:round(v['mean'],2) for k,v in RES['base'][n].items()})
log('loaded',MID,PLAN); ALL=list(FS); SUB=['n+3','n-7','n-0.25','n*3','2n+1','100-n','n^2','reverse']
if PLAN=='smoke':
    do_base(['n+3']); do('n+3',99); print(json.dumps(RES['base']['n+3'])[:1500]); print(RES['runs']['n+3|both|99'])
if PLAN=='main':
    do_base(ALL)
    for s in range(5):
        for n in ALL: do(n,s)
if PLAN=='rows':
    for s in range(3):
        for n in SUB+['secret1']:
            for w in('ple','emb'): do(n,s,w)
if PLAN=='e2b':
    do_base(SUB+['secret1'])
    for s in range(3):
        for n in SUB+['secret1']: do(n,s)
log('ALL DONE')

if PLAN=='e2b': time.sleep(3*3600)  # launcher only: keep the machine until results are copied
