You are preparing a controlled clinical-trial matching robustness experiment.

Select up to MAX_SELECTIONS short, eligibility-relevant spans from the patient note. Return only a JSON object with a `selections` array. Every selection must contain:

- `text`: a verbatim substring that occurs exactly once inside `natural_text`;
- `natural_text`: a verbatim substring that occurs exactly once, contains
  `text`, and can be deleted as a whole without leaving ungrammatical or
  misleading prose;
- `low_quality_text`: a concise replacement for exactly `text` that fits the
  original surrounding grammar and number, removes the original medical value,
  and explicitly says the evidence is reported, unavailable, incomplete,
  uncertain, or unverified;
- `category`: one of `biomarker`, `treatment`, `disease_stage`, `diagnosis`, `performance_status`, `laboratory_value`, `imaging_finding`, `procedure`, `comorbidity`, `symptom`, `demographic`, or `temporal_history`;
- `importance`: either `supporting` or `eligibility_critical`.

Prefer concise facts whose absence or degraded provenance could change eligibility reasoning. Keep `text` limited to the medical value so an explicit placeholder fits its original grammar. Keep `natural_text` to the smallest removable phrase or clause: preserve age, sex, diagnoses, treatments, tests, symptoms, durations, and other facts that are not the selected `text`; never span more than one sentence. If deleting a complete sentence, include its terminal punctuation. Do not rewrite, normalize, summarize, or correct either selected source span. `low_quality_text` must not repeat the value in `text` or invent a substitute value. Check it in the original sentence to avoid duplicated articles such as `a a`, incorrect singular/plural agreement, or altered word order. Do not select patient names, dates used only for de-identification, or unrelated prose. Do not include Markdown fences or any keys other than `selections`.
Preserve the source letter case and punctuation exactly, including capitalization at the beginning of a sentence.

Before selecting a fact, scan the entire note for direct repeats, abbreviations,
and synonymous restatements of that fact. Prefer a fact with one direct mention.
Do not select a repeated fact unless every direct mention can be represented
within MAX_SELECTIONS; indirect clinical clues may remain. Read the full
sentence after mentally substituting `low_quality_text`: articles, verbs,
prepositions, punctuation, and singular/plural agreement must remain natural.
