#!/bin/bash
# Resume from the 26 Nov 02:51 stop. Remaining: prog_rows (19 of 54), then prog_e2b. bash ~/missing-words-work/main/resume.sh
export PATH="$HOME/.local/bin:$PATH" OAUTHLIB_RELAX_TOKEN_SCOPE=1; A(){ perl -e 'alarm shift; exec @ARGV' "$@"; }
W=~/missing-words-work; caffeinate -i -w $$ & n=0; tries=0
up(){ A 150 colab upload -s mw "$1" "/content/$2" </dev/null >/dev/null 2>&1; }
start(){ d=$W/snap/resume_$(date +%m%d_%H%M%S); mkdir -p $d; cp $W/out/* $d/ 2>/dev/null
 A 300 colab new -s mw --gpu L4 </dev/null >/tmp/mw_new 2>&1; A 60 colab sessions </dev/null 2>&1 | grep -q '^\[mw\]' || { echo "$(date +%T) no L4 free, retry in 10 min"; return 2; }
 tries=$((tries+1)); cd $W/out
 for f in qty.py programs.py repair.py ablate_frozen.py glitch_official.py rowfix.py; do up $W/main/$f $f; done
 for f in abl_e4b rep_e4b prog_e4b prog_rows gl_e2b gl_e4b gl_e4b_v2 rep_e2b abl_e2b prog_e2b fix_e4b; do for x in json log; do [ -f $f.$x ] && up $f.$x $f.$x; done; done
 # refuse to start unless the progress files arrived intact
 for f in prog_rows.json prog_rows.log prog_e4b.json prog_e4b.log rep_e2b.json rep_e2b.log gl_e2b.log abl_e2b.log abl_e4b.log rep_e4b.log fix_e4b.json programs.py; do
  s=$f; l=$W/out/$f; [ $f = programs.py ] && l=$W/main/$f
  A 120 colab download -s mw /content/$s /tmp/mw_chk </dev/null >/dev/null 2>&1; cmp -s /tmp/mw_chk $l || { echo "UPLOAD CHECK FAILED $f"; A 60 colab stop -s mw </dev/null; return 1; }; rm -f /tmp/mw_chk; done
 echo "uploads verified"; nohup colab exec -s mw -f $W/main/resume_block.py </dev/null > $W/out/kernel.txt 2>&1 & }
while true; do
 s=$(A 60 colab sessions </dev/null 2>&1); b=$(A 60 colab usage </dev/null 2>&1 | grep -oE "balance: [0-9.]+" | grep -oE "[0-9.]+")
 echo "$(date +%T) balance $b $(echo "$s"|grep -c '^\[mw\]')" | tee -a $W/out/supervise.log
 if [ -n "$b" ] && [ "$(echo "$b < 409.5" | bc)" = 1 ]; then A 60 colab stop -s mw </dev/null; echo "SPEND CAP"; exit 0; fi
 if [ -n "$b" ] && ! echo "$s" | grep -q '^\[mw\]'; then [ $tries -gt 4 ] && { echo "too many restarts"; exit 1; }; waits=$((waits+1)); [ $waits -gt 36 ] && { echo "no L4 for 6 h, giving up"; exit 1; }; start; r=$?; [ $r = 2 ] && { sleep 600; continue; }; [ $r = 1 ] && { sleep 60; continue; }
 else bash $W/main/pull.sh; fi
 n=$((n+1)); [ $((n%5)) = 1 ] && { d=$W/snap/$(date +%m%d_%H%M); mkdir -p $d; cp $W/out/*.json $W/out/*.log $d/; }
 grep -q "ALL DONE" $W/out/prog_e2b.log 2>/dev/null && { bash $W/main/pull.sh; A 60 colab stop -s mw </dev/null; echo "ALL RUNS DONE"; exit 0; }
 sleep 360
done
