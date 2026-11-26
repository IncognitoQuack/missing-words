#!/bin/bash
# snapshots every 30 min, restarts Colab if it disappears, stops at the spend cap
export PATH="$HOME/.local/bin:$PATH" OAUTHLIB_RELAX_TOKEN_SCOPE=1; A(){ perl -e 'alarm shift; exec @ARGV' "$@"; }
W=~/missing-words-work; mkdir -p $W/snap; miss=0; n=0; restarts=0
while true; do
 bash $W/main/pull.sh; bash $W/main/pull2.sh; n=$((n+1))
 [ $((n%5)) = 1 ] && d=$W/snap/$(date +%m%d_%H%M) && mkdir -p $d && cp $W/out/*.json $W/out/*.log $W/out/*.txt $W/main/*.py $W/main/PROTOCOL.md $W/main/FROZEN.sha256 $d/ 2>/dev/null
 b=$(A 60 colab usage </dev/null 2>&1 | grep -oE "balance: [0-9.]+" | grep -oE "[0-9.]+")
 s=$(A 60 colab sessions </dev/null 2>&1)
 echo "$(date +%T) balance $b $(echo "$s" | grep -c '^\[mw')" >> $W/out/supervise.log
 if [ -n "$b" ] && [ "$(echo "$b < 409.5" | bc)" = 1 ]; then A 60 colab stop -s mw </dev/null; A 60 colab stop -s mw2 </dev/null; echo "SPEND CAP reached"; exit 0; fi
 grep -q "ALL DONE" $W/out/prog_e2b.log 2>/dev/null && { echo "ALL PLANNED RUNS DONE"; exit 0; }
 if [ -n "$b" ] && ! echo "$s" | grep -q '^\[mw\]'; then miss=$((miss+1)); else miss=0; fi
 if [ $miss -ge 2 ]; then miss=0; restarts=$((restarts+1)); [ $restarts -gt 4 ] && { echo "too many restarts"; exit 1; }
  echo "$(date +%T) RESTART $restarts" >> $W/out/supervise.log; cp -r $W/out $W/snap/before_restart_$restarts
  A 240 colab new -s mw --gpu L4 </dev/null >/dev/null 2>&1; cd $W/main
  for f in qty.py programs.py repair.py ablate_frozen.py glitch_official.py rowfix.py; do A 120 colab upload -s mw $f /content/$f </dev/null >/dev/null 2>&1; done
  cd $W/out; for f in abl_e4b rep_e4b prog_e4b prog_rows gl_e2b gl_e4b rep_e2b abl_e2b prog_e2b fix_e4b; do for x in json log; do [ -f $f.$x ] && A 120 colab upload -s mw $f.$x /content/$f.$x </dev/null >/dev/null 2>&1; done; done
  nohup colab exec -s mw -f $W/main/resume_block.py </dev/null > $W/out/kernel.txt 2>&1 &
 fi
 sleep 360
done
