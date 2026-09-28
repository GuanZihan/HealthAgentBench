You are analyzing a completed AI-agent rollout for a scientific benchmark.

Produce a neutral, chronological behavioral summary. Consolidate consecutive
steps that pursue one goal into a small number of coherent behavior blocks.
Each block must cover a contiguous range of trajectory step IDs, and blocks
must appear in chronological order.

Focus on:

- what the agent intended to do;
- which data, files, tools, or commands it used;
- how its strategy changed as evidence arrived;
- the observable outcome of each block;
- what the agent ultimately produced and reported.

Use only evidence present in the supplied trajectory. Do not infer hidden
thoughts. Do not classify, diagnose, or highlight meltdown errors in this
version of the analysis. Tool failures may be described factually when they
altered the workflow. Keep titles short, summaries precise, and evidence step
IDs selective. Prefer 4–10 blocks for an ordinary rollout, using more only
when the trajectory has clearly distinct phases.
