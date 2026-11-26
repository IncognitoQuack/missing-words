# Missing Words: what one token's rows can hold in Gemma 4

Gemma 4 E2B and E4B give every token two learnt tables, an input embedding and a per-layer embedding (PLE) row that every decoder layer reads again. This repository asks what one token's rows can hold when every other weight is frozen, under a protocol that was frozen and hashed before the main runs.

**Report:** [`report/report.pdf`](report/report.pdf) (5 pages). **Try it:** [`notebook/missing_words_demo.ipynb`](notebook/missing_words_demo.ipynb), free Colab T4, about 15 minutes.

## Results (official `google/gemma-4-E4B-it` and `E2B-it` weights)

| Pre-registered gate | Outcome | Numbers |
|---|---|---|
| 1. One token holds a rule: ≥16 of 24 rules at 80% on unseen numbers *and* unseen formats; ruleless tables ≤5% (E4B) | **pass** | 19 / 24, mean 82.5% (5 seeds); tables 0.4%, 0.7% |
| 2. Word repair at least doubles the base score in each unseen format, with zero change elsewhere | **fail** | E4B 1.54× / 1.43× / 1.97×; unrelated outputs unchanged (max log-prob change 0.0) |
| 3. Repair beats LoRA given the same word list across scripts by ≥10 points | **fail** | E4B 77% vs 95%; E2B 56% vs 80% |
| 4. Per-layer rows beat input rows, both models | **pass** | +6.4 pts [1.2, 11.5] E4B; +23.3 [15.5, 31.4] E2B |
| 5. Replacing PLE rows by the mean row costs Indian scripts more than English and other Latin-script languages, both models | **pass** | +27.7 / +11.0 pts (E4B); +45.1 / +28.2 (E2B) |

Also reported:

- **A rule needs both tables.** Per-layer rows alone average 50% and the input row alone 58%, against 92% for both.
- **E2B programs are a partial replication:** 3 of 8 rules reach 80%, with a mean of 80%.
- **Broken tokens:** 214 of the 219 tokens fixed in E4B's tokenizer on 3 Nov 2026 still have untrained per-layer rows. A union list holds 361 tokens that fail a repeat test (288 truly broken), and pruning 474 BPE merges fixes all but one.
- **Failed predictions** are listed in the report and in `protocol/PROTOCOL.md`.

## Layout

| Path | What |
|---|---|
| `protocol/` | `PROTOCOL.md`: design, gates, dated amendments, scored outcomes. `FROZEN.sha256`: hashes recorded at each freeze. |
| `src/` | Study code as run on Colab: `programs.py` (S1), `repair.py` (S2), `ablate_frozen.py` (S3), `glitch_official.py` (S4), `rowfix.py` (S5), `classify.py`, and `analyse.py`, which scores every gate from `results/`. `*_frozen.py` are the hashed versions; the others add only launcher hooks, listed in the amendments. |
| `src/exploratory/` | Pilot and exploratory scripts on a 4-bit MLX copy (`GEMMA_E4B_MLX`). Not part of the gates. |
| `ops/` | Colab orchestration: supervision, pulls, resume with byte-checked uploads. |
| `results/` | Raw JSON and logs for every run; `t4/` holds the E2B runs made on a T4. |
| `report/` | LaTeX source, figures and `figs.py` (redraws every figure from `results/`). |
| `notebook/` | One-click demo. |

```bash
python src/analyse.py
```

```bash
python report/figs.py
```

The first rescores every gate from `results/`; the second redraws the figures. Both need numpy, and figures also need matplotlib.

## Limits

Two small models, one family of arithmetic rules and one family of quantity prompts. At most 5 seeds; E2B programs use 3 seeds and 8 rules. The Hindi items have been checked by one native speaker. Ablating a table after training is not the same as training without it. See the report for the rest.
