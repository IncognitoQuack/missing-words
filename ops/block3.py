import subprocess,threading,time
from google.colab import runtime
threading.Thread(target=lambda:(time.sleep(6*3600),runtime.unassign()),daemon=True).start()
open('/content/run.sh','w').write(r'''pip install -q -U transformers accelerate peft datasets 2>&1 | tail -1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P="python -u"; E4=google/gemma-4-E4B-it; E2=google/gemma-4-E2B-it
MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
a(){ $P /content/repair.py $E2 /content/rep_e2b.json e2b > /content/rep_e2b.log 2>&1; $P /content/programs.py $E2 /content/prog_e2b.json e2b > /content/prog_e2b.log 2>&1; }
b(){ $P /content/programs.py $E4 /content/prog_rows.json rows > /content/prog_rows.log 2>&1; }
if [ $MB -gt 60000 ]; then a & b & wait; else a; [ $MB -gt 20000 ] && b; fi
touch /content/ALLDONE
''')
p=subprocess.Popen(['bash','/content/run.sh'],stdout=open('/content/vm.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
while p.poll() is None: time.sleep(60); print(time.strftime('%H:%M'),flush=True)
