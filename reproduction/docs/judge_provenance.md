# LLM-as-a-Judge provenance

## Status

The preserved Judge outputs and C / C* checklists support reproducible
post-hoc analysis.

Exact historical generation provenance is incomplete.

Accordingly:

- preserved-output Judge analysis: reproducible from archived outputs;
- exact historical Judge generation rerun: not certified.

## Judge models

The thesis identifies two scoring models:

### Judge A

- model family: Qwen3-VL-32B-Instruct
- role: Judge A
- exact historical weight/provider revision: not recovered

### Judge B

- model family: Gemma-3-27B-it
- role: Judge B
- exact historical weight/provider revision: not recovered

The repository does not invent missing revision identifiers.

## Machine-readable C and C* checklists

The reproduction artifact preserves the archived checklists under:

`reproduction/prompts/checklists/`

Files:

- `checklist_C_Qwen3VL32B.json`
- `checklist_C_star_Qwen3VL32B.json`
- `checklist_C_Gemma3.json`
- `checklist_C_star_Gemma3.json`

These files preserve the actual checklist item text.

The C* files additionally preserve the archived item weights and semantic
categories.

## Qwen checklist

The Qwen C checklist contains 10 items.

Its C* checklist also contains 10 weighted items.

The preserved C* emphasizes:

- visual grounding / unsupported-detail avoidance;
- temperature / weather;
- style;
- occasion;
- color and mood;
- query-style language;
- clarity / fluency.

The highest preserved Qwen C* weight is assigned to the visual-grounding
criterion that checks whether unsupported visual details are introduced.

## Gemma checklist

The Gemma C checklist contains 15 items.

Its C* checklist contains 15 weighted items.

The preserved C* emphasizes:

- visual grounding;
- overall visual impression;
- temperature / season;
- occasion / activity;
- style;
- query language;
- colors / texture / mood;
- clarity and concision.

The historical C* includes an additional temperature-in-degrees-Celsius
criterion identified as `c16`.

## Preserved formal Judge outputs

The archived handoff contains formal score outputs for the full 35,140
outfit set for both Judge models.

The reproduction configuration points to the original archived outputs
under the senior-student data tree.

These preserved outputs are analyzed by the secondary-analysis reproduction
scripts without claiming that a fresh LLM generation was performed.

## Prompt robustness experiment

The thesis documents the following prompt variants:

- P0: formal scoring prompt;
- P0-R2: repeated execution using the same prompt setting;
- P1: criterion/order variation;
- P2: conservative variation where insufficient or uncertain evidence is
  treated as not satisfying the criterion.

The preserved robustness outputs support recalculation of the reported
robustness statistics.

The robustness sample size is 150 records.

The thesis describes 75 high-score and 75 low-score records.

## Historical prompt limitation

The complete exact historical text files for the Judge scoring prompt
versions P0, P1 and P2 have not been recovered from the archived handoff.

The thesis describes their purpose and evolution, but that description is
not equivalent to the exact executable historical prompt text.

Therefore this repository does not synthesize replacement prompt files and
does not claim exact prompt-level generation reproducibility.

Missing exact historical prompt text is recorded as a provenance gap.

## Historical generation-parameter limitation

The archived evidence does not establish exact historical values for all of
the following generation parameters:

- temperature;
- top_p;
- top_k;
- max_new_tokens;
- repetition penalty;
- precision;
- batch size;
- generation seed.

These values remain unresolved rather than guessed.

## Human audit

The preserved human-audit design uses:

- 30 selected cases;
- 10 low-score cases;
- 10 high-score cases;
- 10 disagreement/conflict cases;
- sampling seed 42;
- 750 total anonymized human judgments.

The human-audit sampling seed 42 is independent of the five model-training
seeds used in the 2026 main reproduction.

## Interpretation

The Judge component therefore has two distinct reproducibility levels:

1. **Preserved-output analysis**:
   numerical Judge outputs, checklists and downstream statistics can be
   inspected and recalculated from preserved artifacts.

2. **Exact historical LLM generation**:
   incomplete because exact model revisions, full P0/P1/P2 executable prompt
   files and complete generation parameters were not recovered.

No missing historical prompt or generation parameter is fabricated in this
artifact.
