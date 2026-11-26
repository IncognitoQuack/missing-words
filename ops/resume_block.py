import subprocess,threading,time
from google.colab import runtime
threading.Thread(target=lambda:(time.sleep(8*3600),runtime.unassign()),daemon=True).start()
open('/content/run6.sh','w').write(r'''pip install -q -U transformers accelerate peft datasets 2>&1 | tail -1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="python -u"; E4=google/gemma-4-E4B-it; E2=google/gemma-4-E2B-it
go(){ o=$1; shift; grep -q "ALL DONE" /content/$o.log 2>/dev/null || "$@" /content/$o.json ${X} >> /content/$o.log 2>&1; }
X= go abl_e4b $P /content/ablate_frozen.py $E4
X=main go rep_e4b $P /content/repair.py $E4
X=main go prog_e4b $P /content/programs.py $E4
X= go abl_e2b $P /content/ablate_frozen.py $E2
X=e2b go rep_e2b $P /content/repair.py $E2
X=rows go prog_rows $P /content/programs.py $E4
X= go gl_e2b $P /content/glitch_official.py $E2
X=e2b go prog_e2b $P /content/programs.py $E2
''')
p=subprocess.Popen(['bash','/content/run6.sh'],stdout=open('/content/vm.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
while p.poll() is None: time.sleep(60); print(time.strftime('%H:%M'),flush=True)
