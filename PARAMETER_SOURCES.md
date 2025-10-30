# Parameter Sources: Set vs Inferred

This document clarifies which parameters are **SET a priori** vs **INFERRED from data** in the Bayesian EVC analysis.

## Overview

| Parameter | Current Implementation | Real Experiment | Method |
|-----------|----------------------|-----------------|--------|
| `evidence_clarity` | **SET** (simulated) | **INFERRED** | DDM, RT, confidence ratings |
| `decision_uncertainty` | **COMPUTED** from evidence_clarity | **INFERRED** | From evidence_clarity or directly |
| `state_uncertainty` | **INFERRED** (Step 2) | **INFERRED** | Bayesian updating from observations |
| `confidence` | **COMPUTED** from uncertainty | **INFERRED** | Confidence ratings, DDM |
| `entropy` | **COMPUTED** from probabilities | **COMPUTED** | Always computed, never set |
| `uncertainty_weight (λ)` | **INFERRED** (fitted) | **INFERRED** | Model fitting (Step 4) |

---

## Detailed Explanation

### 1. Evidence Clarity (`evidence_clarity`)

**Current Implementation (Simulation):**
- **SET a priori** in `utils/data_generator.py`
- Randomly generated values per block (e.g., 0.7-0.9 for low uncertainty, 0.3-0.5 for high uncertainty)
- Formula: `evidence_clarity = np.random.uniform(min, max)`

**In Real Experiments:**
- **INFERRED from behavioral data**
- Methods to infer:
  1. **Drift-Diffusion Model (DDM)**: Estimate drift rate from reaction times and accuracy
     - Higher drift rate → clearer evidence
     - `evidence_clarity ≈ drift_rate / max_drift_rate`
  
  2. **Reaction Time Patterns**: Faster RT → clearer evidence
     - `evidence_clarity ≈ 1 - (RT / max_RT)`
  
  3. **Confidence Ratings**: Direct measure if collected
     - `evidence_clarity ≈ confidence_rating`
  
  4. **Accuracy Patterns**: Higher accuracy → clearer evidence
     - `evidence_clarity ≈ accuracy_rate`

**Code Location:** `utils/data_generator.py` lines 58-68 (currently SET)

---

### 2. Decision Uncertainty (`decision_uncertainty`)

**Current Implementation:**
- **COMPUTED** from `evidence_clarity` (not inferred)
- Formula: `decision_uncertainty = 1 - evidence_clarity`
- This is a simple transformation, not inference

**In Real Experiments:**
- **INFERRED** if evidence_clarity is inferred
- Or **INFERRED directly** from:
  1. **Confidence ratings**: `decision_uncertainty = 1 - confidence`
  2. **DDM entropy**: From drift rate and boundary separation
  3. **RT variability**: More variable RT → higher uncertainty

**Code Location:** `models/bayesian_uncertainty.py` line 96 (COMPUTED)

---

### 3. State Uncertainty (`state_uncertainty`)

**Current Implementation:**
- **INFERRED** using Bayesian updating (Step 2)
- Uses observations (accuracy outcomes) to update beliefs
- Formula: `state_uncertainty = entropy(beliefs) / max_entropy`
- Where beliefs are updated via: `P(state|observation) ∝ P(observation|state) × P(state)`

**In Real Experiments:**
- **INFERRED** the same way (Bayesian updating)
- Uses trial outcomes to update beliefs about which task rule is active
- This is TRUE INFERENCE from data

**Code Location:** `models/bayesian_uncertainty.py` lines 223-286 (INFERRED)

---

### 4. Confidence (`confidence`)

**Current Implementation:**
- **COMPUTED** from uncertainty measures
- Formula: `confidence = 1 / (1 + exp(-5 × (evidence_clarity - 0.5)))`
- Or: `confidence = 1 - combined_uncertainty`

**In Real Experiments:**
- **INFERRED** from:
  1. **Confidence ratings**: Direct measure if collected
  2. **DDM confidence**: From drift rate and decision time
  3. **RT patterns**: Slower RT often indicates lower confidence

**Code Location:** Multiple locations, currently COMPUTED

---

### 5. Entropy

**Current Implementation:**
- **ALWAYS COMPUTED** (never set or inferred)
- Formula: `H(P) = -Σ P(x) × log₂(P(x))`
- This is a measure computed from probability distributions

**In Real Experiments:**
- **ALWAYS COMPUTED** the same way
- Entropy is a mathematical measure, not a parameter to infer

**Code Location:** `models/bayesian_uncertainty.py` lines 136-137 (COMPUTED)

---

### 6. Uncertainty Weight (λ) - KEY PARAMETER

**Current Implementation:**
- **INFERRED** via model fitting (Step 4)
- Optimized to minimize prediction error
- This is what you're TESTING in your PhD project

**In Real Experiments:**
- **INFERRED** the same way (model fitting)
- Estimated from observed control allocation patterns
- This is the MAIN PARAMETER you want to estimate

**Code Location:** `models/bayesian_evc.py` fit() method (INFERRED)

---

## Summary

### What's SET (A Priori):
- `evidence_clarity` - In simulation only (randomly generated)
- `rule_stability` - In simulation only (manipulated per block)
- Likelihood matrices in Bayesian updating (can be inferred too)

### What's COMPUTED (Transformations):
- `decision_uncertainty` = 1 - `evidence_clarity`
- `confidence` = sigmoid(`evidence_clarity`)
- `entropy` = -Σ P(x) × log₂(P(x))

### What's INFERRED (From Data):
- **State uncertainty** - From Bayesian updating using trial outcomes
- **Uncertainty weight (λ)** - From model fitting to observed control
- **Evidence clarity** - Would be inferred in real experiments (DDM, RT, etc.)

---

## For Real Experiments

To adapt this code for real data:

1. **Replace evidence_clarity generation** with inference:
   - Use DDM to estimate from RT/accuracy
   - Use confidence ratings if available
   - Use RT patterns

2. **State uncertainty** already uses inference (Step 2)

3. **Uncertainty weight (λ)** is already inferred (Step 4)

The key insight: Currently `evidence_clarity` is SET for simulation purposes, but in real experiments it should be INFERRED from behavioral data (RT, accuracy, confidence).

