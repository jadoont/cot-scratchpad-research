# Emergent Reasoning in Small Language Models
## Chain-of-Thought vs. Scratchpad Training

**Researcher:** Tayyaba Jadoon | Dickinson College COMP560  
**Supervisors:** Prof. MacCormick, Prof. Goble, Prof. Ferland  
**Status:** In progress 

## Overview
Controlled experiments comparing three training approaches on 
character-level arithmetic using PyTorch and NanoGPT:
- Baseline (direct prediction)
- Chain-of-Thought (explicit reasoning steps)
- Scratchpad (free intermediate computation space)

## Key Finding
Scratchpad-trained models spontaneously developed structured carry 
notation without explicit supervision — early empirical evidence for 
autonomous intermediate reasoning representation.

## Evaluation
Built eval framework measuring accuracy by digit count, carry 
complexity, and OOD generalization across all 3 conditions. 
Identified systematic degradation at higher digit lengths.

## Stack
Python · PyTorch · NanoGPT · HuggingFace · Character-level tokenization
