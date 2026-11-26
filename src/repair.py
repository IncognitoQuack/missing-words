# Official Google weights, HF reference implementation. Usage: python official.py MODEL_ID OUT.json
import sys,os,json,re,random,time,collections,math,torch
import torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0,'/content'); from qty import *
from transformers import AutoTokenizer, AutoModelForMultimodalLM
MID=sys.argv[1]; OUT=sys.argv[2]; RES={'model':MID}; T0=time.time()
def log(*a): print(f'[{(time.time()-T0)/60:5.1f}m]',*a,flush=True)
def save(): json.dump(RES,open(OUT,'w'),ensure_ascii=False,indent=1)
tok=AutoTokenizer.from_pretrained(MID); tok.padding_side='left'
model=AutoModelForMultimodalLM.from_pretrained(MID,dtype=torch.bfloat16,device_map='cuda').eval()
for p in model.parameters(): p.requires_grad_(False)
import transformers; RES['transformers']=transformers.__version__; RES['torch']=torch.__version__
emb=ple=None
for n,m in model.named_modules():
    if n.endswith('language_model.embed_tokens'): emb=m; log(n,tuple(m.weight.shape),'scale',float(m.embed_scale))
    if n.endswith('embed_tokens_per_layer'): ple=m; log(n,tuple(m.weight.shape),'scale',float(m.embed_scale))
NL=ple.weight.shape[1]//256; V=ple.weight.shape[0]
log('loaded',MID,'GPU GB',round(torch.cuda.memory_allocated()/1e9,1))
CONC={'sava':['सवा','sava','sawa'],'saadhe':['साढ़े','साढे','saadhe','sadhe'],'paune':['पौने','paune'],'dedh':['डेढ़','डेढ','dedh'],'dhaai':['ढाई','dhaai','dhai']}
EXTRA={'sava':['सव्वा','સવા','সোয়া','ਸਵਾ','سوا'],'saadhe':['साडे','સાડા','সাড়ে','ਸਾਢੇ','ساڑھے'],'paune':['पावणे','પોણા','পৌনে','ਪੌਣੇ','پونے']}
NAMES=list(CONC); IDS=[tok.convert_tokens_to_ids(f'<unused{200+k}>') for k in range(len(NAMES))]; assert all(isinstance(i,int) and i>100 for i in IDS),IDS
ALT={w:k for k,n in enumerate(NAMES) for w in CONC[n]+EXTRA.get(n,[])}
RX=re.compile(r'\s*(?<![\wऀ-෿؀-ۿ])('+'|'.join(sorted(map(re.escape,ALT),key=len,reverse=True))+r')(?![\wऀ-෿؀-ۿ])')
def chat(u): return tok.apply_chat_template([{'role':'user','content':u}],add_generation_prompt=True,enable_thinking=False,tokenize=False)
CAN=[False]; CANON={w:CONC[n][0] for n in NAMES for w in CONC[n]+EXTRA.get(n,[])}
def canon(u): return re.sub(RX.pattern[3:] if RX.pattern.startswith(r'\s*') else RX.pattern,lambda m:CANON[m.group(1)],u)
def enc(u,patch):
    if CAN[0]: u=canon(u)
    s=chat(u)
    if not patch: return tok(s,add_special_tokens=False)['input_ids']
    out=[];p=0
    for m in RX.finditer(s):
        if s[p:m.start()]: out+=tok(s[p:m.start()],add_special_tokens=False)['input_ids']
        out.append(IDS[ALT[m.group(1)]]); p=m.end()
    return out+tok(s[p:],add_special_tokens=False)['input_ids']
class Row:
    def __init__(s,mod,init): s.mod=mod; s.rows=nn.Parameter(init.float().cuda()); s.on=False; s.lut=torch.full((V+8,),-1,device='cuda'); s.lut[torch.tensor(IDS,device='cuda')]=torch.arange(len(IDS),device='cuda'); mod.register_forward_hook(s.hook)
    def hook(s,m,i,o):
        if not s.on: return o
        k=s.lut[i[0]]; mk=k>=0
        if not mk.any(): return o
        return torch.where(mk.unsqueeze(-1),(s.rows[k.clamp(min=0)]*float(m.embed_scale)).to(o.dtype),o)
def pieces(w): return tok(' '+w,add_special_tokens=False)['input_ids']
def unit(P): P=P.view(P.shape[0],NL,256); return (P/P.norm(dim=-1,keepdim=True)).reshape(P.shape[0],-1)
E0=torch.stack([emb.weight[pieces(CONC[n][0])].float().mean(0) for n in NAMES]); E0=E0/E0.norm(dim=-1,keepdim=True)*1.1
P0=unit(torch.stack([ple.weight[pieces(CONC[n][0])].float().mean(0) for n in NAMES]))
RE,RP=Row(emb,E0),Row(ple,P0)
EOT=tok.convert_tokens_to_ids('<turn|>'); EOS=[1,EOT]; PAD=0
@torch.no_grad()
def gen(qs,patch,mt=56,bs=24):
    RE.on=RP.on=patch; outs=[]
    for i in range(0,len(qs),bs):
        e=[enc(q,patch) for q in qs[i:i+bs]]; L=max(map(len,e)); x=torch.tensor([[PAD]*(L-len(a))+a for a in e],device='cuda'); am=torch.tensor([[0]*(L-len(a))+[1]*len(a) for a in e],device='cuda')
        o=model.generate(input_ids=x,attention_mask=am,max_new_tokens=mt,do_sample=False,pad_token_id=PAD,eos_token_id=EOS)
        outs+=tok.batch_decode(o[:,L:],skip_special_tokens=True)
    RE.on=RP.on=False; return outs
import unicodedata
UW={'सौ':100,'हज़ार':1000,'हजार':1000,'लाख':10**5,'करोड़':10**7,'करोड':10**7,'sau':100,'hazaar':1000,'hazar':1000,'thousand':1000,'thousands':1000,'lakh':10**5,'lakhs':10**5,'crore':10**7,'crores':10**7,'લાખ':10**5,'লাখ':10**5,'লক্ষ':10**5,'ਲੱਖ':10**5,'لاکھ':10**5}
def asc(s): return ''.join(str(unicodedata.digit(c)) if c.isdigit() and not c.isascii() and unicodedata.digit(c,None) is not None else c for c in s).replace('٬',',').replace('٫','.')
def num(s):
    s=asc(s); m=re.search(r'(?:amount_inr|amount|rupees|inr)"?\s*[:=]\s*"?₹?\s*(\d[\d,]*\.?\d*)',s); t=m.group(1) if m else None; end=m.end() if m else 0
    if t is None:
        ms=list(re.finditer(r'\d[\d,]*\.?\d*',s))
        if not ms: return None
        t=ms[-1].group(0); end=ms[-1].end()
    try: v=float(t.replace(',','').rstrip('.'))
    except Exception: return None
    w=re.match(r'\s*([^\W\d_][\w\u0900-\u0DFF]*)',s[end:])
    if w and w.group(1) in UW: v=round(v*UW[w.group(1)],4)
    return int(v) if v==int(v) else v
DD={}
def ask(p): return f"इस राशि को अंकों में लिखिए: {p['text']} रुपये। केवल संख्या लिखिए, और कुछ नहीं।"
def askr(p): return f"Convert this amount to digits: {p['text']} rupaye. Reply with only the number, nothing else."
def tool(p,r=False):
    say=f"Ramesh ko {p['text']} rupaye bhej do." if r else f"रमेश को {p['text']} रुपये भेज दो।"
    return 'You are a payments assistant. Turn the user\'s request into one JSON object {"recipient": string, "amount_inr": number} and output only that JSON.\n\nUser: '+say
FM={'ask(train fmt)':ask,'tool(UNSEEN fmt)':tool,'func(UNSEEN fmt)':lambda p:f'Call exactly one function: send_money(to: str, inr: int). Output only the call.\nUser: "सीता को {p["text"]} रुपये भेजो"','eng-q(UNSEEN fmt)':lambda p:f"My landlord says the rent deposit is {p['text']} रुपये. What is that as a plain integer? Answer with the integer only."}
HELD={4,7,9,11,13,16,19}; allp=phrases(True); tr=[p for p in allp if p['n'] not in HELD]; te=[p for p in allp if p['n'] in HELD]
random.seed(1); random.shuffle(tr); random.shuffle(te)
def rate(ok): return [int(sum(ok)),len(ok)]
def heldout(patch):
    r={}
    for k,f in FM.items():
        o=gen([f(p) for p in te],patch); by=collections.defaultdict(list)
        for t,p in zip(o,te): by[p['frac']].append(num(t)==p['val'])
        r[k]={'all':rate([x for v in by.values() for x in v]),**{a:rate(b) for a,b in by.items()}}
    return r
# ---- never-trained transfer sets
TS={}
q=[];g=[]
for n in range(3,11):
    for w,d in(('sava',F(1,4)),('saadhe',F(1,2)),('paune',F(-1,4))):
        for u,uv in(('lakh',10**5),('hazaar',1000)): q.append(f"Ramesh ko {w} {ROM[n]} {u} rupaye bhej do. How many rupees? Reply with only the number."); g.append(int((n+d)*uv))
TS['romanised']=(q,g,num)
q=[];g=[]
for n in range(3,12):
    for w,d in(('सवा',250),('साढ़े',500),('पौने',-250)): q.append(f"{w} {DEV[n]} किलो आलू में कितने ग्राम होते हैं? केवल संख्या लिखो।"); g.append(n*1000+d)
TS['kg_to_g']=(q,g,num)
q=[];g=[]
for n in range(2,12):
    for w,d in(('सवा',15),('साढ़े',30),('पौने',-15)):
        if w=='साढ़े' and n<3: continue
        q.append(f"{w} {DEV[n]} घंटे में कितने मिनट होते हैं? केवल संख्या लिखो।"); g.append(n*60+d)
TS['hours_to_min']=(q,g,num)
def pt(t):
    m=re.search(r'(\d{1,2})\s*[:：.]\s*(\d{2})',asc(t)); return (int(m.group(1))%12,int(m.group(2))) if m else None
q=[];g=[]
for h in range(2,13):
    for w,m in(('सवा',15),('साढ़े',30),('पौने',-15)):
        if w=='साढ़े' and h<3: continue
        M=h*60+m; q.append(f"ट्रेन {w} {DEV[h]} बजे है। समय को HH:MM में लिखो, और कुछ नहीं।"); g.append((M//60%12,M%60))
TS['clock']=(q,g,pt)
NUM={'mr':['','एक','दोन','तीन','चार','पाच','सहा','सात','आठ','नऊ','दहा'],'gu':['','એક','બે','ત્રણ','ચાર','પાંચ','છ','સાત','આઠ','નવ','દસ'],'bn':['','এক','দুই','তিন','চার','পাঁচ','ছয়','সাত','আট','নয়','দশ'],'ur':['','ایک','دو','تین','چار','پانچ','چھ','سات','آٹھ','نو','دس'],'pa':['','ਇੱਕ','ਦੋ','ਤਿੰਨ','ਚਾਰ','ਪੰਜ','ਛੇ','ਸੱਤ','ਅੱਠ','ਨੌਂ','ਦਸ']}
UN={'mr':('लाख','रुपये'),'gu':('લાખ','રૂપિયા'),'bn':('লাখ','টাকা'),'ur':('لاکھ','روپے'),'pa':('ਲੱਖ','ਰੁਪਏ')}; WD={'mr':('सव्वा','साडे','पावणे'),'gu':('સવા','સાડા','પોણા'),'bn':('সোয়া','সাড়ে','পৌনে'),'ur':('سوا','ساڑھے','پونے'),'pa':('ਸਵਾ','ਸਾਢੇ','ਪੌਣੇ')}
for L in NUM:
    q=[];g=[]
    for n in range(3,11):
        for w,dv in zip(WD[L],(F(1,4),F(1,2),F(-1,4))): q.append(f"Convert to digits: {w} {NUM[L][n]} {UN[L][0]} {UN[L][1]}. Reply with only the number."); g.append(int((n+dv)*10**5))
    TS['lang_'+L]=(q,g,num)
IDI=[("'ढाई आखर प्रेम का, पढ़े सो पंडित होय' किसका दोहा है? एक शब्द में।",['कबीर']),("ढाई दिन का झोंपड़ा किस शहर में है? एक शब्द में।",['अजमेर']),("साढ़े साती किस ग्रह से जुड़ी है? एक शब्द में।",['शनि']),("शतरंज में कौन सा मोहरा ढाई घर चलता है? एक शब्द में।",['घोड़','घोड़ा','घोडा']),("डेढ़ दर्जन में कितनी चीजें होती हैं? केवल संख्या।",['18']),("डेढ़ घंटे में कितने मिनट? केवल संख्या।",['90']),("ढाई घंटे में कितने मिनट? केवल संख्या।",['150']),("सवा घंटे में कितने मिनट? केवल संख्या।",['75']),("'डेढ़ इंच ऊपर' किस तरह का प्रयोग है, मुहावरा या गणना? एक शब्द में।",['मुहावरा']),("'नौ दिन चले अढ़ाई कोस' का अर्थ संक्षेप में बताओ।",['धीम','धीर','कम','आलस'])]
def transfer(patch):
    r={}
    for k,(q,g,f) in TS.items(): r[k]=rate([f(t)==y for t,y in zip(gen(q,patch),g)])
    o=gen([a for a,_ in IDI],patch,mt=60); r['idioms_no_harm']=rate([any(k in asc(t) for k in ks) for t,(_,ks) in zip(o,IDI)]); return r
def rawfail():
    random.seed(0); pick=lambda L,k: random.sample(L,min(k,len(L))); r={}
    for nme,P,rom in(('frac_dev',pick(phrases(True),120),0),('frac_rom',pick(phrases(False),120),1),('whole_dev_2_20',pick(whole(True,range(2,21)),50),0),('whole_rom_2_20',pick(whole(False,range(2,21)),50),1),('whole_dev_21_99',pick(whole(True,range(21,100)),80),0)):
        for fn,f in(('ask',(askr if rom else ask)),('tool',lambda p:tool(p,rom))):
            o=gen([f(p) for p in P],False); by=collections.defaultdict(list)
            for t,p in zip(o,P): by[p['frac']].append(num(t)==p['val'])
            r[nme+'|'+fn]={'all':rate([x for v in by.values() for x in v]),**({a:rate(b) for a,b in by.items()} if 'frac' in nme else {})}
            if nme=='frac_dev' and fn=='ask': RES['examples']=[(p['text'],p['val'],t) for t,p in zip(o,P) if num(t)!=p['val']][:25]
    return r
# ---- general prompts (off-trigger)
from datasets import load_dataset
try:
    D=load_dataset('databricks/databricks-dolly-15k',split='train').shuffle(seed=0); G=[x['instruction'] for x in D if not x['context'] and 20<len(x['instruction'])<200][:160]
except Exception as e: log('dolly',e); G=["Explain photosynthesis in two sentences."]*8
H=["भारत की राजधानी क्या है?","ताजमहल किसने बनवाया?","मुझे चाय बनाने की विधि बताओ।","दो लाख रुपये को अंकों में लिखो।","बारिश पर एक छोटी कविता लिखो।","दिल्ली से मुंबई की दूरी कितनी है?","सूरज पूरब से क्यों निकलता है?","एक प्रेरक कहानी सुनाओ।","कंप्यूटर क्या है?","क्रिकेट के नियम समझाओ।"]
GTR=G[:80]+H[:5]; GEV=G[80:160]+H[5:]; RES['n_general_eval']=len(GEV)
base_gen=gen(GEV,False,mt=40)
@torch.no_grad()
def first_lp(qs): return [torch.log_softmax(model(input_ids=torch.tensor([enc(q,False)],device='cuda')).logits[0,-1].float(),-1).cpu() for q in qs]
base_lp=first_lp(GEV)
def drift(tag):
    n=gen(GEV,False,mt=40); lp=first_lp(GEV); kl=[float((b.exp()*(b-c)).sum()) for b,c in zip(base_lp,lp)]
    return {'outputs_changed':[sum(a!=b for a,b in zip(base_gen,n)),len(GEV)],'kl_mean':sum(kl)/len(kl),'kl_max':max(kl)}
# ---- training data
NM=['रमेश','सीता','अमित','नेहा','राहुल','पूजा']
def ind(v):
    s=str(v)
    if len(s)<=3: return s
    h,t=s[:-3],s[-3:];parts=[]
    while len(h)>2: parts.insert(0,h[-2:]);h=h[:-2]
    if h: parts.insert(0,h)
    return ','.join(parts)+','+t
TRF=[(lambda p,n:ask(p),lambda p,n:str(p['val'])),(lambda p,n:f"{p['text']} को अंकों में लिखो। सिर्फ संख्या दो।",lambda p,n:str(p['val'])),(lambda p,n:f"Write this amount in digits: {p['text']} रुपये. Reply with only the number.",lambda p,n:str(p['val'])),
 (lambda p,n:f"{n} ने {p['text']} रुपये का बिल दिया। बिल कितने का था? भारतीय अंक शैली में लिखो।",lambda p,n:'₹'+ind(p['val'])),(lambda p,n:f'Extract the payment as JSON with keys "to" and "amount" (integer rupees). Message: {n} को {p["text"]} रुपये दे दो',lambda p,n:f'```json\n{{"to": "{n}", "amount": {p["val"]}}}\n```'),
 (lambda p,n:f'Convert to one line of JSON like {{"action": ..., "payee": ..., "rupees": ...}}: "{n} को {p["text"]} रुपये चुकाओ"',lambda p,n:f'{{"action": "pay", "payee": "{n}", "rupees": {p["val"]}}}'),(lambda p,n:f"{p['text']} रुपये = ? (numerals)",lambda p,n:ind(p['val'])),(lambda p,n:f"दुकानदार बोला: यह {p['text']} रुपये का है। How many rupees is that? Number only.",lambda p,n:str(p['val']))]
def mk(patch,bs,ret=None):
    B=[]
    for p in random.sample(tr,bs):
        q,a=random.choice(TRF);n=random.choice(NM);B.append((enc(q(p,n),patch),tok(a(p,n),add_special_tokens=False)['input_ids']+[EOT]))
    if ret:
        for q,o in random.sample(ret,bs): B.append((enc(q,False),o))
    L=max(len(a)+len(b) for a,b in B); X=torch.tensor([a+b+[PAD]*(L-len(a)-len(b)) for a,b in B],device='cuda'); M=torch.tensor([[0]*len(a)+[1]*len(b)+[0]*(L-len(a)-len(b)) for a,b in B],device='cuda'); A=torch.tensor([[1]*(len(a)+len(b))+[0]*(L-len(a)-len(b)) for a,b in B],device='cuda'); return X,M,A
def loss(X,M,A):
    lg=model(input_ids=X[:,:-1],attention_mask=A[:,:-1],use_cache=False).logits; m=M[:,1:].bool(); return Fn.cross_entropy(lg[m].float(),X[:,1:][m])
def train_rows(which,steps=160,lr=1e-3,bs=8,acc=2,seed=0):
    random.seed(seed); RE.rows.data=E0.clone(); RP.rows.data=P0.clone(); ps=[]
    if which in('both','emb'): ps.append(RE.rows)
    if which in('both','ple'): ps.append(RP.rows)
    for p in(RE.rows,RP.rows): p.requires_grad_(False)
    for p in ps: p.requires_grad_(True)
    opt=torch.optim.Adam(ps,lr=lr); t=time.time()
    for s in range(1,steps+1):
        RE.on=RP.on=True; opt.zero_grad()
        for _ in range(acc): l=loss(*mk(True,bs))/acc; l.backward()
        opt.step()
        with torch.no_grad():
            if which!='emb': RP.rows.data=unit(RP.rows.data)
            if which!='ple': RE.rows.data=RE.rows.data/torch.clamp(RE.rows.data.norm(dim=-1,keepdim=True)/1.3,min=1.0)
        if s%40==0: log(which,s,'loss',round(float(l)*acc,4),f'{time.time()-t:.0f}s')
    RE.on=RP.on=False; return time.time()-t

PLAN=sys.argv[3] if len(sys.argv)>3 else 'main'
if os.path.exists(OUT): RES.update(json.load(open(OUT)))
for k in('cond',): RES.setdefault(k,{})
def allev(patch): return {'heldout':heldout(patch),'transfer':transfer(patch)}
def put(k,v): RES['cond'][k]=v; save(); log('DONE',k,[sum(x['all'][0] for x in v['heldout'].values()),sum(x['all'][1] for x in v['heldout'].values())],v['transfer'],v.get('drift'))
def need(k): return k not in RES['cond']
# glossary / few-shot wrappers
GL=("Indian number words: सवा/sava X = X + 0.25; साढ़े/saadhe X = X + 0.5; पौने/paune X = X - 0.25; डेढ़/dedh = 1.5; ढाई/dhaai = 2.5; सवा before a unit alone = 1.25. The fraction applies to the multiplier, e.g. पौने तीन लाख = 2.75 x 100000 = 275000.\n\n")
FSH="Examples: पौने तीन लाख = 275000; सवा पाँच हज़ार = 5250; साढ़े छह सौ = 650; डेढ़ करोड़ = 15000000; पौने बारह हज़ार = 11750; सवा आठ लाख = 825000; साढ़े दस लाख = 1050000; ढाई सौ = 250.\n\n"
PRE=['']; _chat=chat
def chat(u): return _chat(PRE[0]+u)
if 'raw' not in RES:
    RES['raw']=rawfail(); save(); log('RAW',{k:v['all'] for k,v in RES['raw'].items()})
if need('base'): put('base',allev(False))
if need('base+lexicon'): CAN[0]=True; put('base+lexicon',allev(False)); CAN[0]=False
if need('glossary'): PRE[0]=GL; put('glossary',allev(False)); PRE[0]=''
if need('glossary+lexicon'): PRE[0]=GL; CAN[0]=True; put('glossary+lexicon',allev(False)); PRE[0]='';CAN[0]=False
if need('fewshot'): PRE[0]=FSH; put('fewshot',allev(False)); PRE[0]=''
# ---- keep data: other senses, trained values never overlap test values
KEEP=[]
for n in(2,3,5,6,8,10): KEEP+= [(f"{w} {DEV[n]} घंटे में कितने मिनट होते हैं? केवल संख्या लिखो।",str(n*60+d)) for w,d in(('सवा',15),('साढ़े',30),('पौने',-15)) if not(w=='साढ़े' and n<3)]
for n in(3,5,6,8,10): KEEP+=[(f"{w} {DEV[n]} किलो आलू में कितने ग्राम होते हैं? केवल संख्या लिखो।",str(n*1000+d)) for w,d in(('सवा',250),('साढ़े',500),('पौने',-250))]
IDT=["'ढाई अक्षर' का अर्थ क्या है?","डेढ़ साल का बच्चा क्या खाता है?","'सवा सेर' मुहावरे का अर्थ बताओ।","साढ़े साती के उपाय क्या माने जाते हैं?","ढाई किलो का हाथ किस फ़िल्म का संवाद है?","डेढ़ होशियार किसे कहते हैं?","सवा महीने का क्या मतलब है?","साढ़े तीन हाथ ज़मीन का क्या अर्थ है?"]
KO=gen(IDT,False,mt=40); KEEP_ID=[(q,tok(o,add_special_tokens=False)['input_ids'][:40]+[EOT]) for q,o in zip(IDT,KO)]
TSK={k:v for k,v in TS.items()}
def keep_split():  # evaluate kg/hours only on numbers not in KEEP
    out={}
    for k,held in(('hours_to_min',{4,7,9,11}),('kg_to_g',{4,7,9,11})):
        q,g,f=TS[k]; ix=[i for i,s in enumerate(q) if any(f' {DEV[n]} ' in s for n in held)]; out[k+'_heldN']=([q[i] for i in ix],[g[i] for i in ix],f)
    return out
TS.update(keep_split())
_mk=mk
def mk2(patch,bs):
    B=[]
    for _ in range(bs):
        u=random.random()
        if u<.6:
            p=random.choice(tr); q,a=random.choice(TRF);n=random.choice(NM);B.append((enc(q(p,n),patch),tok(a(p,n),add_special_tokens=False)['input_ids']+[EOT]))
        elif u<.85: q,a=random.choice(KEEP); B.append((enc(q,patch),tok(a,add_special_tokens=False)['input_ids']+[EOT]))
        else: q,o=random.choice(KEEP_ID); B.append((enc(q,patch),o))
    L=max(len(a)+len(b) for a,b in B); X=torch.tensor([a+b+[PAD]*(L-len(a)-len(b)) for a,b in B],device='cuda'); M=torch.tensor([[0]*len(a)+[1]*len(b)+[0]*(L-len(a)-len(b)) for a,b in B],device='cuda'); A=torch.tensor([[1]*(len(a)+len(b))+[0]*(L-len(a)-len(b)) for a,b in B],device='cuda'); return X,M,A
def offmax():
    RE.on=RP.on=True
    with torch.no_grad(): d=max(float((torch.log_softmax(model(input_ids=torch.tensor([enc(q,True)],device='cuda')).logits[0,-1].float(),-1).cpu()-b).abs().max()) for q,b in zip(GEV,base_lp))
    RE.on=RP.on=False; return d
S5=range(5) if PLAN=='main' else range(3)
for which in('ple','emb','both'):
    for sd in S5:
        k=f'row_{which}_{sd}'
        if not need(k): continue
        train_rows(which,seed=sd); v=allev(True); v['drift']={'max_abs_logprob':offmax()}; put(k,v)
for sd in range(3):
    k=f'row_ple_keep_{sd}'
    if need(k):
        mk=lambda patch,bs,ret=None: mk2(patch,bs); train_rows('ple',steps=240,seed=sd); mk=_mk; v=allev(True); v['drift']={'max_abs_logprob':offmax()}; put(k,v); torch.save({'ple':RP.rows.data.cpu(),'emb':RE.rows.data.cpu(),'ids':IDS},OUT.replace('.json',f'_keep_{sd}.pt'))
from peft import LoraConfig,get_peft_model
for sd in range(3 if PLAN=='main' else 2):
  for tag in('lora','lora_keep'):
    k=f'{tag}_{sd}'
    if not need(k): continue
    torch.manual_seed(sd); random.seed(sd)
    pm=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=r'.*language_model.*\.(q_proj|v_proj)',lora_dropout=0.0)); ps=[p for p in pm.parameters() if p.requires_grad]
    for p in ps: p.data=p.data.float()
    opt=torch.optim.AdamW(ps,lr=1e-4)
    for s in range(160 if tag=='lora' else 240):
        opt.zero_grad()
        for _ in range(2): l=loss(*(_mk(False,8) if tag=='lora' else mk2(False,8)))/2; l.backward()
        opt.step()
    opt=None; torch.cuda.empty_cache(); v=allev(False); CAN[0]=True; v['with_lexicon']=allev(False); CAN[0]=False; v['drift']=drift(tag); put(k,v)
    model=pm.unload()
    for p in model.parameters(): p.requires_grad_(False)
    del pm,ps; torch.cuda.empty_cache()
log('ALL DONE')
