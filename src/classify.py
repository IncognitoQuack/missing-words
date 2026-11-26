"""Split repeat failures into: broken (output unrelated), script (right sound, wrong script), partial (near copy). Fixed rule: similarity of romanised strings >=0.5 -> related."""
import json,sys,difflib,collections,unicodedata
from unidecode import unidecode
def rom(s): return ''.join(c for c in unidecode(s).lower() if c.isalnum())
def kind(s,o):
    o=o.split('\n')[0] if 'Thinking Process' in o else o
    a,b=rom(s),rom(o[:max(4*len(s),24)])
    if not a: return 'unscored'
    if not b: return 'broken'
    r=difflib.SequenceMatcher(None,a,b).ratio(); r=max(r,difflib.SequenceMatcher(None,a,b[:len(a)+2]).ratio())
    sc=lambda t:{unicodedata.name(c,'?').split()[0] for c in t if c.isalpha()}
    if r>=.5: return 'partial' if sc(s)&sc(o) else 'script'
    return 'broken'
def split(rows): return collections.Counter(kind(r[1],r[2]) for r in rows if not r[3])
if __name__=='__main__':
    for f in sys.argv[1:]:
        d=json.load(open(f)); 
        for k in ('flagged','random','union','google_old','candidates','controls','google219_old_tokenizer'):
            if k in d and d[k]: c=split(d[k]); print(f.split('/')[-1],k,'n',len(d[k]),'fails',sum(c.values()),dict(c))
