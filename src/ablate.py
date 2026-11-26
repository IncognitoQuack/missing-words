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

from datasets import load_dataset
import torch.nn.functional as Fn
langs=['eng_Latn','fra_Latn','deu_Latn','spa_Latn','vie_Latn','swh_Latn','yor_Latn','hin_Latn','rus_Cyrl','arb_Arab','urd_Arab','zho_Hans','jpn_Jpan','kor_Hang','tha_Thai','hin_Deva','mar_Deva','ben_Beng','tam_Taml','tel_Telu','guj_Gujr','pan_Guru','mal_Mlym','kan_Knda']
B={l:{x['link']+'|'+str(x['question_number']):x['flores_passage'] for x in load_dataset('facebook/belebele',l,split='test')} for l in langs}
keys=sorted(set.intersection(*[set(v) for v in B.values()])); random.seed(0); random.shuffle(keys); seen=set(); keys=[k for k in keys if not(k.split('|')[0] in seen or seen.add(k.split('|')[0]))][:60]; log('shared',len(keys)); RES['n_passages']=len(keys)
W=ple.weight; mean=torch.stack([c.float().sum(0) for c in W.split(8192)]).sum(0)/W.shape[0]; perm=torch.randperm(V,generator=torch.Generator().manual_seed(0)).cuda(); MODE=['orig']; cfg=model.config.get_text_config(); CUT=cfg.num_hidden_layers-cfg.num_kv_shared_layers-1; RES['cut_layer']=CUT; s16=float(ple.embed_scale)
lo=torch.zeros(NL*256,dtype=torch.bool,device='cuda'); lo[:CUT*256]=True
def ah(m,i,o):
    k=MODE[0]; mm=(mean*s16).to(o.dtype)
    if k=='orig': return o
    if k=='mean': return mm.expand_as(o)
    if k=='zero': return torch.zeros_like(o)
    if k=='perm': return (Fn.embedding(perm[i[0]],W)*s16).to(o.dtype)
    if k=='mean_lower': return torch.where(lo,mm,o)
    if k=='mean_upper': return torch.where(lo,o,mm)
ple.register_forward_hook(ah); ab={}
with torch.no_grad():
    for md in('orig','mean','mean_lower','mean_upper','perm','zero'):
        MODE[0]=md
        for l in langs:
            per=[]
            for k in keys:
                x=torch.tensor([[2]+tok(B[l][k],add_special_tokens=False)['input_ids']],device='cuda'); per.append([float(Fn.cross_entropy(model(input_ids=x[:,:-1],use_cache=False).logits[0].float(),x[0,1:],reduction='sum')),x.shape[1]-1,len(B[l][k])])
            ab.setdefault(l,{})[md]=per
        RES['ab']=ab; save(); log(md)
for l,r in ab.items():
    o=sum(a[0] for a in r['orig']); n=sum(a[1] for a in r['orig']); log(l,'nll/tok %.3f'%(o/n),' '.join('%s +%.3f/tok (%.0f%%)'%(m,(sum(a[0] for a in r[m])-o)/n,100*(sum(a[0] for a in r[m])-o)/o) for m in r if m!='orig'))
log('ALL DONE')
if 'E4B' in MID:  # launcher only: free the GPU, then run the S5 exploratory repair
    os.execvp('python',['python','-u','/content/rowfix.py',MID,'/content/gl_e4b.json','/content/fix_e4b.json'])
