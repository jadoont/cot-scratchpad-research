# Emergent Reasoning in Small Language Models

## Direct vs Chain-of-Thought vs Scratchpad on Algorithmic Addition

This project studies whether reasoning structure helps small transformer models trained from scratch solve addition tasks and generalize beyond the training distribution.

## Current Research Question

Do explicit reasoning traces (Chain-of-Thought) or free intermediate computation space (Scratchpad) improve out-of-distribution generalization compared to direct prediction?

## Main Result So Far

A key result from the current experiments is that **representation matters significantly**.

Using the standard addition format, the model struggled to learn exact addition reliably.  
After switching to a **reversed-digit representation**, a small character-level transformer achieved:

- **99.2% accuracy on reversed 2-digit addition (Direct)**
- **0.0% accuracy on reversed 3-digit addition (Direct)**

For the **reversed Chain-of-Thought model**:

- **99.6% accuracy on reversed 2-digit addition**
- **0.0% accuracy on reversed 3-digit addition**

This suggests that even when the model learns accurate step-by-step reasoning traces on in-distribution tasks, it still does not generalize the algorithm to longer inputs.

## Interpretation

These results suggest:

- small transformers can learn exact addition when the task representation is made more local
- high in-distribution accuracy does **not** imply algorithmic generalization
- explicit reasoning traces may be learned as fixed templates rather than scalable procedures

## Experiment Conditions

- **Direct** — predict the final answer directly
- **Chain-of-Thought (CoT)** — generate deterministic digit-by-digit reasoning steps plus answer
- **Scratchpad** — generate intermediate reasoning inside a scratch region before the answer

## Current Status

### Completed
- synthetic dataset generation
- direct / CoT / scratch training pipelines
- character-level training pipeline
- evaluation by digit length and carry count
- reversed-digit representation experiments
- reversed direct evaluation
- reversed CoT evaluation

### In Progress
- reversed scratchpad evaluation
- failure mode analysis for carry propagation
- out-of-distribution comparison across reasoning conditions

## Key Findings So Far

### Reversed Direct
- 2-digit: **99.2%**
- 3-digit: **0.0%**

### Reversed CoT
- 2-digit: **99.6%**
- 3-digit: **0.0%**

## Stack

- Python
- PyTorch
- nanoGPT-style training
- character-level tokenization
- synthetic data generation
- research-style evaluation and debugging

## Author

Tayyaba Jadoon, 
Dickinson College — COMP 560 Research
