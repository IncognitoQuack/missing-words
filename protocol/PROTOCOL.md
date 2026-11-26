# Protocol, frozen 25 Nov 2026 before the main runs

Seen before freezing: the pilot (25 Nov 01:00-03:00, one seed, HANDOFF.md) and one smoke run of `programs.py` (rule n+3, seed 99). Seed 99 is excluded. Code hashes are in `FROZEN.sha256`. Amendments are dated and listed at the end.

**Models.** Official `google/gemma-4-E4B-it` and `E2B-it`, bf16, Hugging Face, Colab L4, greedy, thinking off.

**S1 Programs** (`programs.py`). 24 rules + 2 rule-less tables; operands 11-99, 59 train, 30 held out (fixed); 6 trained formats, 3 held-out formats; one token, both tables, all weights frozen, 200 steps; E4B seeds 0-4. Baselines on the same held-out operands and formats: rule written out, rule defined in the prompt, 8 examples in the prompt. Primary: mean accuracy over rules on held-out operands **in held-out formats**. `rows` plan: per-layer rows only vs input row only, 8 rules + 1 table, 3 seeds. E2B: 8 rules + 1 table, 3 seeds.

**S2 Repair** (`repair.py`). Five Hindi fraction words; numbers 4, 7, 9, 11, 13, 16, 19 held out; 8 trained formats; 3 unseen formats. Conditions: base, base + word list, glossary, glossary + word list, 8 examples; per-layer / input / both rows (5 seeds E4B, 3 E2B); per-layer + keep data (3); LoRA and LoRA + keep data (3 / 2), each also **with the same word list**. Gated LoRA is LoRA on trigger and the base model off trigger, so it needs no run. A rule-based normaliser scores 100% on these templates by construction and is reported as such. The parser is new: all Indic digits, and "7.25 crore" counts as right. The pilot parser did neither, so pilot base scores may have been too low.

**S3 Per-layer table ablation** (`ablate.py`). 60 parallel Belebele passages, 24 languages; rows replaced by the mean row (all / layers read by later tokens / the rest), shuffled, zero. Outcome: extra nats as a share of the original loss per passage; bootstrap over passages.

**Gates.**
1. Programs headline: >= 80% (5-seed mean, held-out operands, held-out formats) for at least 16 of 24 rules, and <= 5% for both tables.
2. Repair: at least 2x base on held-out numbers in all three unseen formats (per-layer rows), and exactly zero change on unrelated prompts.
3. Cross-script: per-layer rows beat LoRA *with the word list* by >= 10 points on romanised + 5 languages pooled. Otherwise "comparable" or worse, as it falls.
4. Rows beat input rows: same sign in both models (S2 totals), 95% interval over seeds and items excluding zero.
5. Multilingual memory: mean-row cost share larger for the 8 Indic-script languages than for English and than for the 6 other Latin-script languages (bootstrap interval excludes zero, both models).

Whatever fails is reported as failed.

**Amendments.**
- A1, 25 Nov 20:20: `ablate.py` ran out of GPU memory computing the mean row; now summed in chunks. Same quantity; no ablation result had been seen.

- A2, 25 Nov 20:30: gate 5 says 8 Indic-script languages; the code lists 9 (hi, mr, bn, ta, te, gu, pa, ml, kn). All 9 are used. No ablation result seen.
- A3, 25 Nov 21:00: the Colab machine was reclaimed at about 33 minutes. Saved conditions kept; the rest resumes from the same code on a new L4. Order changed only.
- A4, written 25 Nov 21:00 before any official-weights result: S4 (`glitch_official.py`, coded 20:45), designed after the 4-bit result. Claims: flagged tokens fail the repeat test at >= 5x the rate of random unflagged tokens on official weights; the per-layer detector finds more failing tokens in its top 400 than every embedding detector tested; merge pruning repairs >= 90%. Exploratory on E4B, confirmatory on E2B.

- A5, 25 Nov 21:10, before any official-weights result: S5 exploratory row repair (`rowfix.py`), launched from the end of `ablate.py` (measurement code unchanged; frozen copy kept).

- A6, 25 Nov 21:25: official E4B S4 counts were logged, then the JSON save crashed (int64). Gates on E4B: claim 1 pass (334/1662 against 6/3972), claim 3 pass (332/334), claim 2 FAIL (embedding kurtosis 208 > per-layer 191 in top 400; Magikarp better on 4-bit: 341 against 260 in top 500). Script fixed, Magikarp and a union list added; reruns on E2B then E4B. rowfix.py uses the 391 tokens failing on the 4-bit copy and re-tests them on official weights.

- Note, 25 Nov 21:14: times in A1-A6 were estimates and some run ahead of the clock (A3/A4 were written at about 20:58-21:00, A5 at about 21:03, A6 at about 21:12). The order is right.
- A7, 25 Nov 21:22: the rowfix launch did not fire (ablate.py had already started); it now launches from the top of programs.py (measurement code unchanged, frozen copy kept). S3 E4B result: gate 5 holds on E4B; E2B pending.
- A8, 25 Nov 21:30: owner raised the Colab limit to 45 units. E2B S3, S4, S5 and S1 also run on a T4 (bf16 is emulated there) to save time; the planned L4 E2B runs stay queued and are the ones of record.
- A9, 25 Nov 21:46: T4 E2B S4 seen (842/7002 flagged fail, 77/3885 random unflagged, 804/842 repaired). Its rival lists were wrong: unused tokens were not excluded before ranking (0 of 0 picks). Fixed for the L4 runs.
- A10, 25 Nov 21:55: classify.py (romanised similarity < 0.5 = broken) added after seeing that E2B also copies words into another script. Reported alongside, not instead of, the original S4 counts. Known errors in both directions; final lists get a second, differently worded test.
- E4B S2 scored 25 Nov 22:28: gate 2 FAILS (ratio to base in unseen formats: tool 1.54, func 1.43, eng 1.97; needed 2.0; locality exact). Gate 3 FAILS (LoRA + word list 0.946 cross-script against 0.771). Gate 4 E4B sign positive (0.867 against 0.803); E2B pending.
- A11, 25 Nov 22:45: E2B S2, E2B S1 and the rows study move to a second, faster machine with the frozen code; results are copied to the L4, whose resume logic skips them. rowfix E4B seen: per-layer rows 0.836 and input rows 0.930 on the held-out code format (broken 0.125, pruning 0.945); spelling 0.047 / 0.125 (pruning 0.609).
- A11 note, 26 Nov 01:50: the swap in A11 obtained a T4 again, not a faster machine. E2B S2 therefore runs on a T4 with the frozen code; hardware is reported with each result.
- S1 E4B scored 26 Nov 01:50: gate 1 PASSES (19 of 24 rules >= 0.80 in unseen formats, mean 0.825; tables 0.004 and 0.007). Failing: n+0.37 0.57, n-0.6 0.77, n*1.3 0.68, 150-n 0.42, n mod 7 0.05. Borderline passes: n-20 0.81, n^2 0.81. Seed 0-2 results had been looked at while the fixed 5-seed run continued. A written rule in the prompt scores 0.97+ on all but one rule (0.82).
- 26 Nov 02:32: L4 E2B runs complete. S3 E2B L4 repeat: Indic-English +45.0 [40.4,49.6], Indic-other Latin +28.1 [25.5,...]: gate 5 PASSES on both models. S2 E2B (rows on T4, part of keep/LoRA on L4; frozen code): gate 2 passes on E2B alone (tool 3.0, func 3.0, eng 2.3x) but FAILS overall because E4B failed; gate 3 FAILS (0.563 against LoRA + word list 0.798); gate 4 PASSES (ple - emb: E4B +6.4 points [1.2,11.5], E2B +23.3 [15.5,31.4]; bootstrap over seeds and items, script g4.py written after the data). E2B rows harm idioms (0.60 -> 0.37). Second machine stopped: it was repeating work already done on the L4.
- 26 Nov 02:51: run paused at the owner's request; prog_rows 19 of 54 runs, prog_e2b not started. Each run is seeded by itself, so resuming changes nothing. A run in progress was discarded and will be repeated.
- 26 Nov 12:42: resumed 11:10 on a new L4 (uploads byte-checked, continued at run 20); all planned runs done; Colab stopped, balance 426.84 (26.1 of 45 used).
- Rows study (E4B, 8 rules + table, 3 seeds, unseen formats): per-layer rows alone mean 0.50, input rows alone 0.58, both (main run, 5 seeds) 0.92. Per-layer rows alone score 0 on one of the three unseen formats for most rules. Neither table alone carries the rule across formats; together they do. Tables stay <= 4%.
- S1 E2B (8 rules + table, 3 seeds): 3 of 8 rules >= 0.80, mean 0.80, table 0.015. Weaker than E4B; reported as a partial replication (gate 1 is defined on E4B).
