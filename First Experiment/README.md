# First Experiment

## Goal

This experiment is the first priority validation for the Nested Learning idea:

- **Baseline**: standard `Transformer`
- **Proposed variant**: `Transformer + CMS` (`Continuum Memory System`)
- **Task**: `continual learning`

The main purpose is to check whether adding a multi-timescale memory mechanism to a standard Transformer backbone can improve learning under distribution shifts and reduce catastrophic forgetting.

## Hypothesis

A Transformer with CMS should be better than a plain Transformer on continual learning because:

1. It separates knowledge into multiple memory levels with different update frequencies.
2. It can preserve older information in slower-updating components.
3. It should adapt to new tasks without overwriting all previously learned knowledge.

## What this experiment should prove

We want to test whether the CMS design helps the model:

- learn new tasks faster
- forget old tasks less
- maintain better average performance over a task sequence
- generalize better under sequential domain or class shifts

## Recommended experimental setup

### Backbone

Use the same Transformer backbone for both models.

- `Transformer`: standard FFN blocks
- `Transformer + CMS`: replace the FFN part with a CMS-style memory module, or add a CMS block in place of the vanilla MLP/FFN

### Training protocol

Train on a sequence of tasks, one after another, without mixing future tasks into earlier stages.

This can be done with:

- class-incremental learning
- domain-incremental learning
- sequential language adaptation

### Suggested datasets

A good first choice is a class-incremental text classification benchmark such as:

- `CLINC`
- `Banking`
- `DBpedia`

If a simpler prototype is needed, use a small sequential text classification setup first, then scale up to one of the above datasets.

## Baselines

At minimum, compare:

- `Transformer`
- `Transformer + CMS`

If time allows, also add:

- `Transformer + AdamW` as the standard optimizer setup
- `Transformer + CMS + AdamW` if CMS is only a structural change
- `EWC` or another continual learning baseline

## Metrics

Track at least the following:

- **Average accuracy** across all tasks
- **Forgetting** on earlier tasks
- **Forward transfer** to new tasks
- **Backward transfer** if applicable
- **Per-task accuracy** after each training stage

## Expected outcome

If the Nested Learning idea is useful, `Transformer + CMS` should:

- outperform the plain Transformer in average continual learning performance
- retain more performance on earlier tasks
- show a better stability-plasticity tradeoff

## Implementation notes

For the first version, keep the implementation simple:

- do not implement the full Hope architecture yet
- keep attention unchanged
- only modify the memory/FFN side first
- keep the training loop and optimizer standard

This experiment is mainly to validate the effect of **multi-timescale memory** under continual learning.

## Next steps

If this first experiment is promising, the next experiments can add:

1. more memory levels
2. self-modifying updates
3. long-context reasoning tasks
4. optimizer-side improvements

## Status

- [ ] Define exact dataset and task split
- [ ] Implement baseline Transformer
- [ ] Implement Transformer + CMS
- [ ] Run continual learning evaluation
- [ ] Compare metrics and analyze forgetting
