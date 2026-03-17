# Current Findings

## 1. Standard representation was hard to learn
In the original setup, the model produced plausible-looking but incorrect numeric outputs. It learned the format but not exact addition.

## 2. Reversed representation fixed learnability
Switching to reversed-digit addition made the task much easier for the model.

## 3. Reversed direct result
- 2-digit: 99.2%
- 3-digit: 0.0%

The model solves seen problems but fails completely on longer ones.

## 4. Reversed Chain-of-Thought result
- 2-digit: 99.6%
- 3-digit: 0.0%

Even with reasoning steps, the model does not generalize.

## 5. Next step
Test scratchpad training to see if free reasoning improves generalization.