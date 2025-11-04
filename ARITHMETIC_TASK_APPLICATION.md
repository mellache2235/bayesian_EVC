# Applying Bayesian EVC to Arithmetic Tasks in Children

## Overview

Your Bayesian EVC framework is **perfectly suited** for modeling cognitive control in arithmetic tasks! This application addresses:

1. ✅ **Varying problem difficulty** - Different uncertainty levels
2. ✅ **Learning from past trials** - Temporal dynamics via HGF
3. ✅ **Individual differences** - Children vary in math ability
4. ✅ **Cognitive control allocation** - Effort on hard vs. easy problems

---

## The Arithmetic Task Scenario

### **Typical Setup:**

```
Children (ages 7-12) solve arithmetic problems:
- Easy: 3 + 5 = ?
- Medium: 17 + 28 = ?
- Hard: 234 + 567 = ?

Measured:
- Reaction time (RT)
- Accuracy
- Confidence ratings (if available)
- Pupil dilation (if available)
```

### **Research Questions:**

1. **How does problem difficulty affect control allocation?**
   - Do children exert more control on harder problems?
   
2. **Does uncertainty from past performance affect current control?**
   - After getting several problems wrong, do they try harder?
   
3. **How do children learn to allocate control efficiently?**
   - Do they learn which problems need more effort?
   
4. **Individual differences in control allocation?**
   - Some children work hard on all problems (anxious?)
   - Others give up on hard problems (low persistence?)

---

## How Bayesian EVC Maps to Arithmetic Tasks

### **Traditional EVC Variables:**

| EVC Variable | Arithmetic Task Mapping |
|--------------|------------------------|
| **Reward** | Points/praise for correct answers |
| **Accuracy** | Probability of getting it right (depends on difficulty) |
| **Control** | Mental effort, working memory engagement |
| **Effort Cost** | Metabolic cost, cognitive fatigue |

### **Bayesian EVC Extensions:**

| Bayesian Variable | Arithmetic Task Mapping |
|------------------|------------------------|
| **Decision Uncertainty** | How hard is this problem? (difficulty level) |
| **State Uncertainty** | Am I good at these problems? (confidence from past trials) |
| **Volatility** | Is my performance getting better/worse? (learning trajectory) |
| **Control (measured)** | RT, pupil dilation, strategy choice |

---

## Data Generation for Arithmetic Tasks

### **What We Need to Simulate:**

```python
For each child, each problem:
- Problem: "17 + 28 = ?"
- Difficulty: Easy/Medium/Hard (1-5 scale)
- Correct answer: 45
- Child's ability: Math skill level
- Past performance: Recent successes/failures
- Control allocated: Effort exerted
- Outcome: Correct/incorrect + RT
```

---

## Complete Implementation

### **Step 1: Generate Arithmetic Task Data**

```python
import numpy as np
import pandas as pd


class ArithmeticTaskGenerator:
    """
    Generate arithmetic task data for children
    
    Simulates:
    - Problems of varying difficulty
    - Children with different math abilities
    - Control allocation based on difficulty and uncertainty
    - Learning from past trials
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def generate_arithmetic_problem(self, difficulty):
        """
        Generate an arithmetic problem of given difficulty
        
        Args:
            difficulty: 1 (easy) to 5 (hard)
            
        Returns:
            dict with problem info
        """
        if difficulty == 1:
            # Easy: single-digit addition
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 10)
            operation = '+'
            
        elif difficulty == 2:
            # Medium-easy: two-digit addition
            a = np.random.randint(10, 50)
            b = np.random.randint(10, 50)
            operation = '+'
            
        elif difficulty == 3:
            # Medium: two-digit addition or subtraction
            a = np.random.randint(20, 100)
            b = np.random.randint(10, 50)
            operation = np.random.choice(['+', '-'])
            if operation == '-' and b > a:
                a, b = b, a  # Ensure positive result
                
        elif difficulty == 4:
            # Hard: three-digit addition or two-digit multiplication
            if np.random.rand() > 0.5:
                a = np.random.randint(100, 500)
                b = np.random.randint(100, 500)
                operation = '+'
            else:
                a = np.random.randint(10, 30)
                b = np.random.randint(10, 30)
                operation = '*'
                
        else:  # difficulty == 5
            # Very hard: three-digit operations
            a = np.random.randint(100, 999)
            b = np.random.randint(100, 999)
            operation = np.random.choice(['+', '-', '*'])
            if operation == '-' and b > a:
                a, b = b, a
            if operation == '*':
                a = a // 100  # Keep products manageable
                b = b // 100
        
        # Compute answer
        if operation == '+':
            answer = a + b
        elif operation == '-':
            answer = a - b
        else:  # '*'
            answer = a * b
        
        # Problem string
        problem_string = f"{a} {operation} {b} = ?"
        
        return {
            'problem': problem_string,
            'operand1': a,
            'operand2': b,
            'operation': operation,
            'correct_answer': answer,
            'difficulty': difficulty
        }
    
    def generate_child_data(self, 
                           child_id,
                           n_trials=100,
                           math_ability=0.7,
                           uncertainty_tolerance=0.5,
                           persistence=0.7):
        """
        Generate data for one child
        
        Args:
            child_id: Child identifier
            n_trials: Number of problems
            math_ability: Base math skill (0-1)
            uncertainty_tolerance: How much they respond to uncertainty
            persistence: How hard they try on difficult problems
        """
        trials = []
        
        # HGF-like state tracking
        belief_about_ability = math_ability  # Self-assessment
        uncertainty_about_ability = 0.5  # How uncertain about own ability
        
        for trial in range(n_trials):
            # Select difficulty (random but weighted by past performance)
            if trial < 10:
                # Early trials: mix of easy and medium
                difficulty = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            else:
                # Later trials: adapt to child's ability
                if belief_about_ability > 0.7:
                    difficulty = np.random.choice([2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.2])
                elif belief_about_ability > 0.5:
                    difficulty = np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2])
                else:
                    difficulty = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            
            # Generate problem
            problem = self.generate_arithmetic_problem(difficulty)
            
            # ============================================
            # COMPUTE CONTROL ALLOCATION (Bayesian EVC)
            # ============================================
            
            # Reward (points for correct answer)
            reward = difficulty * 10  # Harder problems worth more points
            
            # Expected accuracy (depends on difficulty and ability)
            base_accuracy = math_ability * (1.1 - 0.2 * difficulty)
            base_accuracy = np.clip(base_accuracy, 0.1, 0.95)
            
            # Decision uncertainty (from problem difficulty)
            problem_uncertainty = (difficulty - 1) / 4  # Scale 0-1
            
            # State uncertainty (from recent performance)
            state_uncertainty = uncertainty_about_ability
            
            # Combined uncertainty
            total_uncertainty = 0.6 * problem_uncertainty + 0.4 * state_uncertainty
            
            # BAYESIAN EVC: Control allocation
            baseline_control = 0.3
            
            # Traditional EVC component
            expected_value = reward * base_accuracy
            evc_component = expected_value * 0.1
            
            # Uncertainty component (KEY!)
            uncertainty_component = uncertainty_tolerance * total_uncertainty * 0.4
            
            # Persistence (individual difference)
            difficulty_boost = persistence * (difficulty / 5) * 0.3
            
            # Total control
            control_signal = baseline_control + evc_component + \
                           uncertainty_component + difficulty_boost
            
            # Add noise
            control_signal = control_signal + np.random.normal(0, 0.1)
            control_signal = np.clip(control_signal, 0, 1)
            
            # ============================================
            # SIMULATE PERFORMANCE
            # ============================================
            
            # Control improves accuracy
            actual_accuracy = base_accuracy + control_signal * 0.2
            actual_accuracy = np.clip(actual_accuracy, 0.05, 0.98)
            
            # Determine outcome
            correct = np.random.rand() < actual_accuracy
            
            # Reaction time (higher control → slower, harder problem → slower)
            base_rt = 1000 + difficulty * 500  # ms
            control_rt_increase = control_signal * 1000  # More control = slower
            rt = base_rt + control_rt_increase + np.random.normal(0, 200)
            rt = max(rt, 300)  # Minimum RT
            
            # Confidence (inverse of uncertainty)
            confidence = 1 - total_uncertainty
            confidence = np.clip(confidence, 0.1, 0.9)
            
            # ============================================
            # UPDATE BELIEFS (for next trial)
            # ============================================
            
            # Update belief about own ability
            prediction_error = (1 if correct else 0) - belief_about_ability
            learning_rate = 0.1 + 0.1 * uncertainty_about_ability  # Higher when uncertain
            
            belief_about_ability += learning_rate * prediction_error
            belief_about_ability = np.clip(belief_about_ability, 0.1, 0.9)
            
            # Update uncertainty
            if correct:
                # Success reduces uncertainty
                uncertainty_about_ability *= 0.95
            else:
                # Failure increases uncertainty
                uncertainty_about_ability = min(uncertainty_about_ability * 1.05, 0.8)
            
            # ============================================
            # STORE TRIAL DATA
            # ============================================
            
            trials.append({
                # Identifiers
                'child_id': child_id,
                'trial': trial + 1,
                
                # Problem info
                'problem': problem['problem'],
                'difficulty': difficulty,
                'correct_answer': problem['correct_answer'],
                
                # Child characteristics
                'math_ability': math_ability,
                'uncertainty_tolerance': uncertainty_tolerance,
                'persistence': persistence,
                
                # Cognitive variables
                'problem_uncertainty': problem_uncertainty,
                'state_uncertainty': state_uncertainty,
                'total_uncertainty': total_uncertainty,
                'belief_about_ability': belief_about_ability,
                
                # Control
                'control_signal': control_signal,
                
                # Outcome
                'correct': int(correct),
                'rt': rt,
                'confidence': confidence,
                
                # For analysis
                'reward': reward,
                'expected_accuracy': base_accuracy
            })
        
        return pd.DataFrame(trials)
    
    def generate_dataset(self, 
                        n_children=30,
                        n_trials_per_child=100,
                        age_range=(7, 12)):
        """
        Generate full dataset of children doing arithmetic
        
        Args:
            n_children: Number of children
            n_trials_per_child: Problems per child
            age_range: (min_age, max_age)
        """
        all_data = []
        
        print(f"Generating arithmetic task data for {n_children} children...")
        
        for child_id in range(n_children):
            # Simulate child characteristics
            age = np.random.uniform(age_range[0], age_range[1])
            
            # Math ability increases with age (roughly)
            base_ability = 0.3 + (age - age_range[0]) / (age_range[1] - age_range[0]) * 0.4
            math_ability = base_ability + np.random.normal(0, 0.15)
            math_ability = np.clip(math_ability, 0.2, 0.9)
            
            # Individual differences
            uncertainty_tolerance = np.random.uniform(0.2, 0.8)  # How much uncertainty affects them
            persistence = np.random.uniform(0.4, 0.9)  # How hard they try on hard problems
            
            # Generate data
            child_data = self.generate_child_data(
                child_id=child_id,
                n_trials=n_trials_per_child,
                math_ability=math_ability,
                uncertainty_tolerance=uncertainty_tolerance,
                persistence=persistence
            )
            
            # Add age
            child_data['age'] = age
            
            all_data.append(child_data)
            
            if (child_id + 1) % 10 == 0:
                print(f"  Generated {child_id + 1}/{n_children} children...")
        
        # Combine
        dataset = pd.concat(all_data, ignore_index=True)
        
        print(f"✓ Generated {len(dataset)} trials total")
        print(f"  Age range: {dataset['age'].min():.1f} - {dataset['age'].max():.1f} years")
        print(f"  Mean accuracy: {dataset['correct'].mean():.2%}")
        print(f"  Difficulty distribution:")
        for diff in range(1, 6):
            pct = (dataset['difficulty'] == diff).mean()
            print(f"    Level {diff}: {pct:.1%}")
        
        return dataset


# ============================================
# USAGE
# ============================================

if __name__ == '__main__':
    # Generate data
    generator = ArithmeticTaskGenerator(seed=42)
    
    data = generator.generate_dataset(
        n_children=30,
        n_trials_per_child=100,
        age_range=(7, 12)
    )
    
    # Save
    import os
    os.makedirs('data/arithmetic', exist_ok=True)
    data.to_csv('data/arithmetic/arithmetic_task_data.csv', index=False)
    
    print("\n✓ Saved to: data/arithmetic/arithmetic_task_data.csv")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    
    print(f"\nOverall:")
    print(f"  Children: {data['child_id'].nunique()}")
    print(f"  Total trials: {len(data)}")
    print(f"  Mean accuracy: {data['correct'].mean():.2%}")
    print(f"  Mean RT: {data['rt'].mean():.0f} ms")
    
    print(f"\nBy difficulty:")
    for diff in range(1, 6):
        diff_data = data[data['difficulty'] == diff]
        print(f"  Level {diff}: Accuracy = {diff_data['correct'].mean():.2%}, "
              f"RT = {diff_data['rt'].mean():.0f} ms")
    
    print(f"\nBy age:")
    for age in [7, 8, 9, 10, 11, 12]:
        age_data = data[(data['age'] >= age) & (data['age'] < age + 1)]
        if len(age_data) > 0:
            print(f"  Age {age}: Accuracy = {age_data['correct'].mean():.2%}, "
                  f"N = {age_data['child_id'].nunique()} children")
```

---

## Applying Bayesian EVC to Arithmetic Data

### **Model Specification:**

```python
# Traditional EVC
Control = (Reward × Expected_Accuracy - Cost(Control)) / (2 × Cost_Weight)

# Bayesian EVC for Arithmetic
Control = Baseline + 
          (Reward × Expected_Accuracy + λ × Uncertainty) / (2 × Cost_Weight)
          
Where:
- Reward = Points for problem (higher for harder problems)
- Expected_Accuracy = Child's estimated probability of success
- Uncertainty = Problem difficulty + self-doubt from past failures
- λ = Uncertainty weight (individual difference)
```

### **Temporal Component (HGF):**

```python
# Track child's belief about their own ability over time
HGF tracks:
- Belief about math ability (μ₂)
- Uncertainty about ability (σ₂²)
- Volatility in performance (μ₃)

After each trial:
- Success → belief increases, uncertainty decreases
- Failure → belief decreases, uncertainty increases
- Streak of failures → volatility increases → higher learning rate
```

---

## Research Questions You Can Answer

### **1. Does Problem Difficulty Affect Control?**

**Analysis:**
```python
# Compare control on easy vs. hard problems
easy_control = data[data['difficulty'] == 1]['control_signal'].mean()
hard_control = data[data['difficulty'] == 5]['control_signal'].mean()

print(f"Control on easy problems: {easy_control:.3f}")
print(f"Control on hard problems: {hard_control:.3f}")

# Statistical test
from scipy import stats
t, p = stats.ttest_ind(
    data[data['difficulty'] == 1]['control_signal'],
    data[data['difficulty'] == 5]['control_signal']
)
print(f"Difference: t = {t:.2f}, p = {p:.4f}")
```

**Expected finding:** Higher control on harder problems (if λ > 0)

---

### **2. Does Past Performance Affect Current Control?**

**Analysis:**
```python
# Add "recent success rate" feature
data['recent_success'] = data.groupby('child_id')['correct'].rolling(
    window=5, min_periods=1
).mean().reset_index(0, drop=True)

# Correlation
corr = data['recent_success'].corr(data['control_signal'])
print(f"Correlation(recent_success, control): {corr:.3f}")

# Regression
import statsmodels.api as sm

X = sm.add_constant(data[['recent_success', 'difficulty']])
y = data['control_signal']
model = sm.OLS(y, X).fit()
print(model.summary())
```

**Expected finding:** After failures, control increases (compensatory effort)

---

### **3. Individual Differences in Uncertainty Sensitivity?**

**Analysis:**
```python
# Fit Bayesian EVC per child, extract λ

lambdas = []
for child_id in data['child_id'].unique():
    child_data = data[data['child_id'] == child_id]
    
    # Fit model
    model = BayesianEVC()
    results = model.fit(child_data)
    
    lambdas.append({
        'child_id': child_id,
        'lambda': results['uncertainty_weight'],
        'age': child_data['age'].iloc[0],
        'ability': child_data['math_ability'].iloc[0]
    })

lambdas_df = pd.DataFrame(lambdas)

# Does λ correlate with age?
corr_age = lambdas_df['lambda'].corr(lambdas_df['age'])
print(f"Correlation(λ, age): {corr_age:.3f}")

# Does λ correlate with ability?
corr_ability = lambdas_df['lambda'].corr(lambdas_df['ability'])
print(f"Correlation(λ, ability): {corr_ability:.3f}")
```

**Potential findings:**
- Older children: Lower λ (less affected by uncertainty)
- Higher ability: Lower λ (more confident)
- Anxious children: Higher λ (over-respond to uncertainty)

---

### **4. Learning Trajectory: How Does Control Evolve?**

**Analysis:**
```python
# Fit temporal model
from models.bayesian_evc_temporal import BayesianEVC_Temporal

model = BayesianEVC_Temporal()
results = model.fit(data)

# Plot uncertainty evolution for one child
child_data = data[data['child_id'] == 0]
predictions, uncertainty_traj, volatility_traj = model.predict_control_sequential(
    child_data
)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Uncertainty over trials
axes[0].plot(uncertainty_traj)
axes[0].set_ylabel('Uncertainty about Ability')
axes[0].set_title('Child Learning: Uncertainty Decreases Over Trials')

# Control over trials
axes[1].plot(child_data['control_signal'].values, label='Observed')
axes[1].plot(predictions, label='Predicted', linestyle='--')
axes[1].set_xlabel('Trial')
axes[1].set_ylabel('Control')
axes[1].legend()
axes[1].set_title('Control Allocation: Predicted vs. Observed')

plt.tight_layout()
plt.show()
```

**Expected finding:** Uncertainty decreases as child learns, control becomes more efficient

---

## Clinical/Educational Applications

### **1. Identifying Math Anxiety**

**High λ children:**
- Over-allocate control due to uncertainty
- Effortful even on easy problems
- May need anxiety intervention

```python
# Identify high-λ children
high_lambda = lambdas_df[lambdas_df['lambda'] > lambdas_df['lambda'].quantile(0.75)]
print(f"High uncertainty sensitivity children: {len(high_lambda)}")
print("These children may have math anxiety")
```

---

### **2. Personalized Difficulty Adjustment**

**Use model to select optimal difficulty:**

```python
def select_next_problem(child_model, child_state):
    """
    Select problem difficulty that maximizes learning
    
    Too easy: No challenge, no learning
    Too hard: High uncertainty, gives up
    Just right: Moderate uncertainty, optimal control
    """
    best_difficulty = None
    max_engagement = 0
    
    for difficulty in range(1, 6):
        # Simulate control for this difficulty
        predicted_control = child_model.predict(
            reward=difficulty * 10,
            uncertainty=estimate_uncertainty(difficulty, child_state)
        )
        
        # Engagement = high control but not too stressful
        engagement = predicted_control * (1 - stress_level(predicted_control))
        
        if engagement > max_engagement:
            max_engagement = engagement
            best_difficulty = difficulty
    
    return best_difficulty
```

---

### **3. Detecting Learning Plateaus**

**Use HGF volatility to detect when learning stalls:**

```python
# High volatility = inconsistent performance = struggling
struggling_periods = volatility_traj > volatility_traj.mean() + volatility_traj.std()

print(f"Trials where child is struggling: {np.where(struggling_periods)[0]}")
print("→ Provide intervention during these trials!")
```

---

## Expected Results

### **With Your Lab's Real Data:**

**Hypotheses:**

1. **H1: Uncertainty increases control**
   - Expected: λ > 0 (p < 0.001)
   - Children allocate more effort on uncertain problems

2. **H2: Past failures increase current control**
   - Expected: Positive correlation
   - Compensatory effort after mistakes

3. **H3: Older children more efficient**
   - Expected: Age negatively correlates with λ
   - Better calibration of control to difficulty

4. **H4: Individual differences in λ**
   - Expected: High variance in λ
   - Some children over-control (anxious?)
   - Some under-control (impulsive?)

---

## Publication Potential

### **Target Journals:**

1. **Developmental Science**
   - "Bayesian modeling of cognitive control development in arithmetic"
   
2. **Journal of Educational Psychology**
   - "Individual differences in effort allocation during math problem-solving"
   
3. **Child Development**
   - "How children learn to allocate cognitive resources: A computational account"
   
4. **Cognitive Development**
   - "Uncertainty and control in mathematical cognition"

---

## Next Steps

### **Week 1: Generate Pilot Data**
```bash
python generate_arithmetic_data.py
```

### **Week 2: Fit Models**
```bash
python fit_bayesian_evc_arithmetic.py
```

### **Week 3: Analyze Results**
- Extract λ parameters
- Test hypotheses
- Visualize findings

### **Week 4: Collect Real Data**
- Use simulation to design experiment
- Determine sample size needed
- Plan interventions based on model

---

## Summary

### **Yes! Your Bayesian EVC framework is PERFECT for arithmetic tasks:**

✅ **Varying difficulty** → Different uncertainty levels
✅ **Learning from past** → HGF temporal dynamics
✅ **Individual differences** → Hierarchical Bayesian modeling
✅ **Control allocation** → RT, strategies, effort

**This could be a high-impact application of your framework in educational neuroscience!** 📚🧠

Want me to generate the arithmetic dataset and fitting scripts for you?


