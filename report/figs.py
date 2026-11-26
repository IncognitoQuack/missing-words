import json,io,contextlib,os,numpy as np,matplotlib
matplotlib.use('pdf'); import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()): exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','src','analyse.py')).read())
plt.rcParams.update({'font.family':'Helvetica Neue','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e','axes.linewidth':.6,'pdf.fonttype':42})
BLUE,ORANGE,AQUA,GREY,INK='#2a78d6','#eb6834','#1baf7a','#a3a29c','#0b0b0b'
N={}
# ---------- F1 repair
fig,axs=plt.subplots(1,2,figsize=(6.8,2.3),sharex=True)
for ax,f,t in zip(axs,('rep_e4b.json','rep_e2b.json'),('Gemma 4 E4B','Gemma 4 E2B')):
    with contextlib.redirect_stdout(io.StringIO()): C=repair(f)
    rows=[('base','No change',GREY),('glossary','Glossary in prompt',GREY),('fewshot','8 examples in prompt',GREY),('row_emb','Input rows',AQUA),('row_ple','Per-layer rows',BLUE),('row_both','Both tables',BLUE),('lora','LoRA r8 (all text)',ORANGE)]
    for i,(k,lab,c) in enumerate(rows[::-1]):
        v=np.array([x['tot'] for x in C[k]]); m=v.mean(); ax.barh(i,m,color=c,height=.62)
        if len(v)>1: ax.plot([v.min(),v.max()],[i,i],color=INK,lw=.8)
        ax.text(v.max()+.015,i,f'{100*m:.0f}%',va='center',fontsize=7,color=INK)
        N[f'{f}:{k}']=[round(float(m),3),len(v)]
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows[::-1]]); ax.set_xlim(0,1.08); ax.set_title(t,fontsize=8.5,loc='left',color=INK)
    ax.set_xlabel('held-out amounts answered correctly'); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
fig.tight_layout(); fig.savefig('fig_repair.pdf'); plt.close(fig)
# ---------- F2 programs
d=json.load(open(O+'prog_e4b.json')); R={}
for k,v in d['runs'].items():
    n,w,s=k.split('|'); R.setdefault(n,[]).append(v['held_new']['mean'])
rules=[n for n in R if not n.startswith('secret')]; rules.sort(key=lambda n:np.mean(R[n]))
fig,ax=plt.subplots(figsize=(6.8,2.5)); x=np.arange(len(rules)+2)
names=['secret1','secret2']+rules
for i,n in enumerate(names):
    v=np.array(R[n]); ax.plot([i,i],[v.min(),v.max()],color=BLUE,lw=.8,alpha=.6); ax.plot(i,v.mean(),'o',ms=4.5,color=BLUE,mec='white',mew=.6,zorder=3)
    b=d['base'].get(n,{})
    if 'explicit_new' in b: ax.plot(i,b['explicit_new']['mean'],'s',ms=3.5,color=GREY,zorder=2)
    if 'fewshot_new' in b: ax.plot(i,b['fewshot_new']['mean'],'^',ms=3.5,color=ORANGE,zorder=2)
ax.axhline(.8,color='#52514e',lw=.6,ls=(0,(3,3))); ax.text(len(names)-.5,.81,'gate 80%',ha='right',va='bottom',fontsize=6.5,color='#52514e')
ax.set_xticks(x); ax.set_xticklabels(['table A','table B']+rules,rotation=60,ha='right',fontsize=6.8); ax.set_ylim(-.03,1.05)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0)); ax.set_ylabel('unseen numbers, unseen formats')
from matplotlib.lines import Line2D
ax.legend([Line2D([],[],marker='o',color=BLUE,ls='',ms=4.5),Line2D([],[],marker='s',color=GREY,ls='',ms=3.5),Line2D([],[],marker='^',color=ORANGE,ls='',ms=3.5)],['one trained token (mean, range of 5 seeds)','rule written in the prompt','8 worked examples in the prompt'],frameon=False,fontsize=6.8,loc='lower right',bbox_to_anchor=(1,.08))
fig.tight_layout(); fig.savefig('fig_programs.pdf'); plt.close(fig)
N['prog_e4b']={n:round(float(np.mean(R[n])),3) for n in names}
# ---------- F3 which table
r=json.load(open(O+'prog_rows.json'))['runs']; S={}
for k,v in r.items(): n,w,s=k.split('|'); S.setdefault((n,w),[]).append(v['held_new']['mean'])
sub=['n+3','n-7','n-0.25','n*3','2n+1','100-n','n^2','reverse']
fig,ax=plt.subplots(figsize=(6.8,1.9)); W=.26
for j,(w,lab,c) in enumerate((('emb','input rows only',AQUA),('ple','per-layer rows only','#9fc3ee'),('both','both tables',BLUE))):
    vals=[np.mean(S[(n,w)]) if w!='both' else np.mean(R[n]) for n in sub]
    ax.bar(np.arange(len(sub))+(j-1)*W,vals,W*.92,color=c,label=lab); N[f'rows_{w}']=round(float(np.mean(vals)),3)
ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub); ax.set_ylim(0,1.05); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0))
ax.set_ylabel('unseen formats'); ax.legend(frameon=False,ncol=3,fontsize=6.8,loc='upper center',bbox_to_anchor=(.5,1.22))
fig.tight_layout(); fig.savefig('fig_tables.pdf'); plt.close(fig)
# ---------- F4 languages
IND=['hin_Deva','mar_Deva','ben_Beng','tam_Taml','tel_Telu','guj_Gujr','pan_Guru','mal_Mlym','kan_Knda']
def shares(f):
    ab=json.load(open(O+f))['ab']; return {l:np.mean([a[0]/b[0]-1 for a,b in zip(ab[l]['mean'],ab[l]['orig'])]) for l in ab}
A,B=shares('abl_e4b.json'),shares('abl_e2b.json'); langs=sorted(A,key=lambda l:A[l]+B[l])
fig,ax=plt.subplots(figsize=(6.8,2.3))
for i,l in enumerate(langs):
    ax.plot([i,i],[A[l],B[l]],color='#d6d5d0',lw=1,zorder=1)
    ax.plot(i,A[l],'o',ms=4.5,color=BLUE if l in IND else '#9fc3ee',mec='white',mew=.6,zorder=3); ax.plot(i,B[l],'D',ms=3.8,color=ORANGE if l in IND else '#f3b394',mec='white',mew=.6,zorder=3)
ax.set_xticks(range(len(langs))); ax.set_xticklabels([l.split('_')[0]+'\n'+l.split('_')[1] for l in langs],fontsize=6.2)
for t,l in zip(ax.get_xticklabels(),langs): t.set_fontweight('bold' if l in IND else 'normal')
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1,decimals=0)); ax.set_ylabel('extra loss, rows replaced by mean row')
ax.legend([Line2D([],[],marker='o',color=BLUE,ls='',ms=4.5),Line2D([],[],marker='D',color=ORANGE,ls='',ms=3.8)],['E4B','E2B'],frameon=False,fontsize=7,loc='upper left',title='Indian scripts in bold, darker marks',title_fontsize=6.5)
fig.tight_layout(); fig.savefig('fig_languages.pdf'); plt.close(fig)
N['abl_e4b']={l:round(float(A[l]),3) for l in langs}; N['abl_e2b']={l:round(float(B[l]),3) for l in langs}
json.dump(N,open('numbers.json','w'),indent=1)
