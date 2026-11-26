import subprocess,os,threading,time
from google.colab import runtime
def guard():
    time.sleep(8*3600); runtime.unassign()
threading.Thread(target=guard,daemon=True).start()
open('/content/run5.sh','w').write('''pip install -q -U transformers accelerate peft datasets 2>&1 | tail -1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="python -u"; E4=google/gemma-4-E4B-it; E2=google/gemma-4-E2B-it
$P /content/glitch_official.py $E4 /content/gl_e4b.json > /content/gl_e4b.log 2>&1
$P /content/ablate.py $E4 /content/abl_e4b.json > /content/abl_e4b.log 2>&1
$P /content/repair.py $E4 /content/rep_e4b.json main > /content/rep_e4b.log 2>&1
$P /content/programs.py $E4 /content/prog_e4b.json main > /content/prog_e4b.log 2>&1
$P /content/glitch_official.py $E2 /content/gl_e2b.json > /content/gl_e2b.log 2>&1
$P /content/ablate.py $E2 /content/abl_e2b.json > /content/abl_e2b.log 2>&1
$P /content/repair.py $E2 /content/rep_e2b.json e2b > /content/rep_e2b.log 2>&1
$P /content/programs.py $E4 /content/prog_rows.json rows > /content/prog_rows.log 2>&1
$P /content/programs.py $E2 /content/prog_e2b.json e2b > /content/prog_e2b.log 2>&1
touch /content/ALLDONE
''')
p=subprocess.Popen(['bash','/content/run5.sh'],stdout=open('/content/vm.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
while p.poll() is None: time.sleep(60); print(time.strftime('%H:%M'),flush=True)
