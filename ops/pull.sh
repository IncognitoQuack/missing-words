export PATH="$HOME/.local/bin:$PATH" OAUTHLIB_RELAX_TOKEN_SCOPE=1; cd ~/missing-words-work/out
for f in gl_e4b_v2 fix_e4b gl_e4b gl_e2b abl_e4b rep_e4b prog_e4b abl_e2b rep_e2b prog_rows prog_e2b; do for x in json log; do perl -e 'alarm 90; exec @ARGV' colab download -s mw /content/$f.$x $f.$x.new </dev/null >/dev/null 2>&1 && mv $f.$x.new $f.$x; done; done; date > last_pull
