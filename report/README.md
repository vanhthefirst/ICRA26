# Reports

Four write-ups, in the order they were produced. **`prompt_baselines/` is the
current one** — it carries the numbers to quote.

| directory | what it covers | status |
|---|---|---|
| `prompt_baselines/` | The three π₀.₅ baselines on the 228-row benchmark: published-number reproduction, explicit prompts, ambiguous prompts. Two pages. | **current** |
| `pi05_baseline/` | The first sketch-free π₀.₅ baseline, 114 scenes at 3 rollouts each (34.5%). | superseded by `prompt_baselines/` |
| `validation_suites/` | How the three validation suites were built and gated. | current, different subject |
| `initial_section2/` | The earlier proof-of-concept phase, kept as the record of that work. | historical |

## Why `pi05_baseline/` is still here

Its 34.5% is the same measurement `prompt_baselines/` reports as baseline 2, at
3 rollouts per scene instead of 14 and under the old vocabulary, where
"ambiguous" meant a scene with several candidates rather than a caption that
names nothing. It is kept because the numbers in it were real and are cited in
`IMPLEMENTATION_PLAN.md`, not because it should be read first. Nothing in it is
wrong; it is just narrower.

## Building any of them

Each directory holds its own `.tex`. `prompt_baselines/` additionally reads
`tables.tex`, which is generated — never hand-edited:

```bash
python scripts/analyze_baselines.py \
    --arms explicit=pi05_explicit_532 ambiguous=pi05_ambiguous_532 \
    --tables report/prompt_baselines/tables.tex \
    --figdir report/prompt_baselines/figures
cd report/prompt_baselines && latexmk -pdf report.tex
```

A table showing `---` means that arm has not been run yet. See
`RUNBOOK_BASELINES.md`.
