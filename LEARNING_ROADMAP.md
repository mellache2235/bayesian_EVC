# Learning Roadmap: Self-Study Guide for Bayesian EVC Project

## Overview

This guide provides a structured learning path to understand all the mathematical, theoretical, and psychological concepts in this project. The order is designed to build foundational knowledge before tackling advanced topics.

**Estimated total time:** 4-8 weeks (depending on prior background)

---

## Your Project in One Sentence

**"We're extending the Expected Value of Control framework to include Bayesian uncertainty, predicting that people allocate more cognitive control when they're uncertain because uncertainty reduction is valuable."**

---

## Learning Path Overview

```
FOUNDATIONS (Week 1-2)
    ↓
├─ Basic Probability & Statistics
├─ Bayesian Inference Basics
└─ Cognitive Psychology Fundamentals

CORE CONCEPTS (Week 2-3)
    ↓
├─ Cognitive Control
├─ Decision-Making Models
└─ Uncertainty in Cognition

COMPUTATIONAL MODELS (Week 3-4)
    ↓
├─ Drift Diffusion Model
├─ Reinforcement Learning
└─ Hierarchical Gaussian Filter

ADVANCED TOPICS (Week 4-6)
    ↓
├─ Expected Value of Control
├─ Bayesian Cognitive Models
└─ Hierarchical Bayesian Modeling

PROJECT-SPECIFIC (Week 6-8)
    ↓
├─ Your Bayesian EVC Extension
├─ Model Fitting & Evaluation
└─ Computational Psychiatry Applications
```

---

# PHASE 1: FOUNDATIONS (Week 1-2)

## 1.1 Basic Probability & Statistics

### **Why You Need This:**
- Understand uncertainty quantification
- Interpret model outputs (R², correlation, RMSE)
- Grasp Bayesian inference

### **Core Concepts:**
- [ ] Probability distributions (Normal, Bernoulli, etc.)
- [ ] Mean, variance, standard deviation
- [ ] Conditional probability: P(A|B)
- [ ] Bayes' theorem: P(H|D) = P(D|H)×P(H) / P(D)
- [ ] Correlation vs. causation
- [ ] R² and model fit metrics

### **Resources:**

#### **📚 Books (Pick One):**

1. **"Statistics Done Wrong"** by Alex Reinhart
   - **Why:** Short, accessible, focuses on common mistakes
   - **Time:** 2-3 days
   - **Level:** Beginner
   - **Link:** https://www.statisticsdonewrong.com/ (free online)

2. **"Seeing Theory"** (Interactive)
   - **Why:** Visual, interactive probability and statistics
   - **Time:** 1 week
   - **Level:** Beginner
   - **Link:** https://seeing-theory.brown.edu/

#### **🎥 Videos:**

1. **3Blue1Brown - Bayes Theorem**
   - **Link:** https://www.youtube.com/watch?v=HZGCoVF3YvM
   - **Time:** 15 minutes
   - **Why:** Best visual explanation of Bayes' theorem

2. **StatQuest - Statistics Fundamentals**
   - **Link:** https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9
   - **Time:** 3-4 hours total
   - **Why:** Clear, simple explanations with visuals

#### **✍️ Practice:**

```python
# Try these exercises in Python:

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. Generate and visualize distributions
normal_data = np.random.normal(0, 1, 1000)
plt.hist(normal_data, bins=30)
plt.show()

# 2. Compute basic statistics
print(f"Mean: {np.mean(normal_data)}")
print(f"Std: {np.std(normal_data)}")

# 3. Bayes' theorem example
# P(Disease|Positive) = P(Positive|Disease) × P(Disease) / P(Positive)
p_disease = 0.01  # 1% have disease
p_pos_given_disease = 0.95  # 95% sensitivity
p_pos_given_no_disease = 0.05  # 5% false positive

p_pos = p_pos_given_disease * p_disease + p_pos_given_no_disease * (1 - p_disease)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos

print(f"P(Disease|Positive) = {p_disease_given_pos:.3f}")
```

---

## 1.2 Bayesian Inference Basics

### **Why You Need This:**
- Core framework for your project
- Understand belief updating
- Grasp uncertainty quantification

### **Core Concepts:**
- [ ] Prior, likelihood, posterior
- [ ] Belief updating
- [ ] Posterior = Prior × Likelihood
- [ ] Predictive distributions
- [ ] Uncertainty vs. variability

### **Resources:**

#### **📚 Books (Pick One):**

1. **"Bayesian Statistics the Fun Way"** by Will Kurt
   - **Why:** Accessible, intuitive, practical
   - **Time:** 1 week
   - **Level:** Beginner
   - **Chapters:** 1-10

2. **"Think Bayes"** by Allen Downey
   - **Why:** Python-based, hands-on
   - **Time:** 1 week
   - **Level:** Beginner-Intermediate
   - **Link:** https://greenteapress.com/wp/think-bayes/ (free)

#### **🎥 Videos:**

1. **Veritasium - Bayes Theorem**
   - **Link:** https://www.youtube.com/watch?v=R13BD8qKeTg
   - **Time:** 20 minutes
   - **Why:** Real-world intuition

2. **Rasmus Bååth - Bayesian Data Analysis Tutorial**
   - **Link:** https://www.youtube.com/watch?v=3OJEae7Qb_o
   - **Time:** 45 minutes
   - **Why:** Practical introduction

#### **✍️ Practice:**

```python
# Bayesian coin flip example

import numpy as np
import matplotlib.pyplot as plt

# Prior: Uniform belief about coin bias
theta = np.linspace(0, 1, 100)  # Possible coin biases
prior = np.ones_like(theta)  # Uniform prior
prior = prior / prior.sum()  # Normalize

# Observe: 7 heads out of 10 flips
n_heads = 7
n_tails = 3

# Likelihood: P(data | theta)
likelihood = theta**n_heads * (1-theta)**n_tails

# Posterior: P(theta | data)
posterior = prior * likelihood
posterior = posterior / posterior.sum()

# Plot
plt.figure(figsize=(10, 4))
plt.plot(theta, prior, label='Prior', linestyle='--')
plt.plot(theta, likelihood/likelihood.sum(), label='Likelihood', linestyle=':')
plt.plot(theta, posterior, label='Posterior', linewidth=2)
plt.xlabel('Coin Bias (θ)')
plt.ylabel('Probability Density')
plt.legend()
plt.title('Bayesian Inference: Coin Flip')
plt.show()

# Posterior mean and uncertainty
posterior_mean = np.sum(theta * posterior)
posterior_std = np.sqrt(np.sum((theta - posterior_mean)**2 * posterior))
print(f"Posterior mean: {posterior_mean:.3f} ± {posterior_std:.3f}")
```

---

## 1.3 Cognitive Psychology Fundamentals

### **Why You Need This:**
- Understand what cognitive control is
- Know the psychological phenomena you're modeling
- Connect to real behavior

### **Core Concepts:**
- [ ] Attention and working memory
- [ ] Cognitive control and executive function
- [ ] Decision-making processes
- [ ] Metacognition (thinking about thinking)
- [ ] Speed-accuracy tradeoffs

### **Resources:**

#### **📚 Books (Pick One):**

1. **"Thinking, Fast and Slow"** by Daniel Kahneman
   - **Why:** Accessible, foundational, engaging
   - **Time:** 2 weeks
   - **Level:** General audience
   - **Chapters:** Focus on Part 1 (Two Systems)

2. **"Cognition"** by Ashcraft & Radvansky (Textbook)
   - **Why:** Comprehensive, structured
   - **Time:** 2-3 weeks
   - **Level:** Undergraduate
   - **Chapters:** 1-3, 13-14

#### **📄 Papers (Essential):**

1. **"The Magical Number Seven"** - George Miller (1956)
   - **Why:** Classic on working memory limits
   - **Time:** 30 minutes
   - **Link:** Google Scholar

2. **"Attention and Effort"** - Daniel Kahneman (1973)
   - **Why:** Foundational work on cognitive effort
   - **Time:** Chapter 1 (1 hour)

#### **🎥 Videos:**

1. **Crash Course Psychology - Cognition**
   - **Link:** https://www.youtube.com/watch?v=u9yXu57ExZk
   - **Time:** 10 minutes
   - **Why:** Quick overview

2. **Yale Open Courses - Introduction to Psychology**
   - **Link:** https://oyc.yale.edu/psychology/psyc-110
   - **Time:** Lectures 8-10 (3 hours)
   - **Why:** In-depth, high quality

---

# PHASE 2: CORE CONCEPTS (Week 2-3)

## 2.1 Cognitive Control

### **Why You Need This:**
- This is what your model predicts!
- Understand the dependent variable
- Connect to neural mechanisms

### **Core Concepts:**
- [ ] What is cognitive control?
- [ ] Anterior cingulate cortex (ACC) function
- [ ] Conflict monitoring
- [ ] Effort and motivation
- [ ] Control allocation strategies

### **Resources:**

#### **📄 Papers (Must Read):**

1. **"Toward an integrative theory of anterior cingulate cortex function"** - Botvinick et al. (2001)
   - **Why:** Foundational conflict monitoring theory
   - **Time:** 2 hours
   - **Link:** Google Scholar
   - **Focus:** Conflict detection, control signals

2. **"The neural basis of cognitive control"** - Miller & Cohen (2001)
   - **Why:** Comprehensive review
   - **Time:** 3 hours
   - **Link:** Google Scholar

3. **"Mechanisms of cognitive control"** - Egner (2017)
   - **Why:** Modern synthesis
   - **Time:** 2 hours
   - **Link:** Google Scholar

#### **🎥 Videos:**

1. **Amitai Shenhav - Cognitive Control**
   - **Search:** "Amitai Shenhav cognitive control" on YouTube
   - **Time:** 1 hour
   - **Why:** From the EVC creator himself

#### **✍️ Practice:**

```python
# Simulate a Stroop task (classic cognitive control task)

import numpy as np
import matplotlib.pyplot as plt

def stroop_task(n_trials=100, control_level=0.5):
    """
    Simulate Stroop task performance
    
    control_level: 0-1, how much control is allocated
    """
    results = []
    
    for trial in range(n_trials):
        # Random trial type
        congruent = np.random.rand() > 0.5
        
        if congruent:
            # Congruent: Easy, less control needed
            base_accuracy = 0.95
            base_rt = 400  # ms
        else:
            # Incongruent: Hard, more control needed
            base_accuracy = 0.70
            base_rt = 600  # ms
        
        # Control improves accuracy and RT on incongruent trials
        if not congruent:
            accuracy = base_accuracy + control_level * 0.25
            rt = base_rt - control_level * 150
        else:
            accuracy = base_accuracy
            rt = base_rt
        
        # Add noise
        correct = np.random.rand() < accuracy
        rt_actual = rt + np.random.normal(0, 50)
        
        results.append({
            'congruent': congruent,
            'correct': correct,
            'rt': rt_actual,
            'control': control_level
        })
    
    return results

# Compare low vs. high control
low_control = stroop_task(control_level=0.2)
high_control = stroop_task(control_level=0.8)

# Analyze
def analyze_stroop(results):
    congruent = [r for r in results if r['congruent']]
    incongruent = [r for r in results if not r['congruent']]
    
    print(f"Congruent accuracy: {np.mean([r['correct'] for r in congruent]):.3f}")
    print(f"Incongruent accuracy: {np.mean([r['correct'] for r in incongruent]):.3f}")
    print(f"Congruent RT: {np.mean([r['rt'] for r in congruent]):.0f} ms")
    print(f"Incongruent RT: {np.mean([r['rt'] for r in incongruent]):.0f} ms")

print("LOW CONTROL:")
analyze_stroop(low_control)
print("\nHIGH CONTROL:")
analyze_stroop(high_control)
```

---

## 2.2 Decision-Making Under Uncertainty

### **Why You Need This:**
- Understand how uncertainty affects decisions
- Connect to your uncertainty manipulation
- Grasp confidence and metacognition

### **Core Concepts:**
- [ ] Decision confidence
- [ ] Uncertainty types (aleatory vs. epistemic)
- [ ] Risk vs. ambiguity
- [ ] Metacognition
- [ ] Confidence-accuracy relationship

### **Resources:**

#### **📄 Papers (Must Read):**

1. **"Decision confidence and uncertainty in diffusion models"** - Ratcliff & Starns (2009)
   - **Why:** Links uncertainty to decision models
   - **Time:** 2 hours
   - **Link:** Google Scholar

2. **"Metacognition in human decision-making"** - Yeung & Summerfield (2012)
   - **Why:** Comprehensive review of confidence
   - **Time:** 2 hours
   - **Link:** Google Scholar

3. **"Neural systems responding to degrees of uncertainty"** - Hsu et al. (2005)
   - **Why:** Neural basis of uncertainty
   - **Time:** 1.5 hours
   - **Link:** Google Scholar

#### **🎥 Videos:**

1. **Hakwan Lau - Metacognition and Consciousness**
   - **Search:** YouTube
   - **Time:** 1 hour
   - **Why:** Leading researcher on confidence

---

## 2.3 Uncertainty in Cognition

### **Why You Need This:**
- Central to your project!
- Understand different uncertainty types
- Learn how brain tracks uncertainty

### **Core Concepts:**
- [ ] Perceptual uncertainty
- [ ] State uncertainty
- [ ] Volatility (environmental uncertainty)
- [ ] Precision (inverse uncertainty)
- [ ] Uncertainty propagation

### **Resources:**

#### **📄 Papers (Must Read):**

1. **"Uncertainty and anticipation in anxiety"** - Grupe & Nitschke (2013)
   - **Why:** Clinical relevance of uncertainty
   - **Time:** 2 hours
   - **Link:** Google Scholar

2. **"Learning the value of information"** - Behrens et al. (2007)
   - **Why:** How uncertainty affects learning
   - **Time:** 2 hours
   - **Link:** Google Scholar

3. **"Computational mechanisms of curiosity"** - Gottlieb & Oudeyer (2018)
   - **Why:** Uncertainty drives exploration
   - **Time:** 1.5 hours
   - **Link:** Google Scholar

---

# PHASE 3: COMPUTATIONAL MODELS (Week 3-4)

## 3.1 Drift Diffusion Model (DDM)

### **Why You Need This:**
- Standard model of decision-making
- Used in your project for confidence
- Foundation for understanding evidence accumulation

### **Core Concepts:**
- [ ] Evidence accumulation
- [ ] Decision boundaries
- [ ] Drift rate and diffusion noise
- [ ] Speed-accuracy tradeoff
- [ ] RT distributions

### **Resources:**

#### **📄 Papers (Must Read):**

1. **"The diffusion decision model"** - Ratcliff & McKoon (2008)
   - **Why:** Comprehensive tutorial
   - **Time:** 3 hours
   - **Link:** Google Scholar
   - **Note:** This is dense but essential

2. **"A tutorial on fitting the diffusion model"** - Vandekerckhove & Tuerlinckx (2008)
   - **Why:** Practical implementation
   - **Time:** 2 hours
   - **Link:** Google Scholar

#### **🎥 Videos:**

1. **Roger Ratcliff - Diffusion Model Tutorial**
   - **Search:** YouTube
   - **Time:** 1 hour
   - **Why:** From the creator

#### **✍️ Practice:**

```python
# Simple DDM simulation

import numpy as np
import matplotlib.pyplot as plt

def simulate_ddm(drift_rate=0.3, boundary=1.0, noise=1.0, dt=0.001, max_time=5.0):
    """
    Simulate one trial of drift diffusion model
    
    Returns: choice (1 or 0), reaction_time, trajectory
    """
    t = 0
    evidence = 0
    trajectory = [evidence]
    
    while abs(evidence) < boundary and t < max_time:
        # Evidence accumulation
        evidence += drift_rate * dt + noise * np.sqrt(dt) * np.random.randn()
        trajectory.append(evidence)
        t += dt
    
    choice = 1 if evidence > 0 else 0
    rt = t
    
    return choice, rt, trajectory

# Simulate multiple trials
n_trials = 100
results = [simulate_ddm(drift_rate=0.5) for _ in range(n_trials)]

# Analyze
choices = [r[0] for r in results]
rts = [r[1] for r in results]

print(f"Accuracy: {np.mean(choices):.3f}")
print(f"Mean RT: {np.mean(rts):.3f} s")

# Plot example trajectories
plt.figure(figsize=(10, 6))
for i in range(10):
    _, _, traj = results[i]
    plt.plot(traj, alpha=0.5)
plt.axhline(1.0, color='r', linestyle='--', label='Upper boundary')
plt.axhline(-1.0, color='r', linestyle='--', label='Lower boundary')
plt.xlabel('Time steps')
plt.ylabel('Evidence')
plt.title('DDM Trajectories')
plt.legend()
plt.show()
```

---

## 3.2 Reinforcement Learning Basics

### **Why You Need This:**
- Understand learning from rewards
- Connect to value-based decision-making
- Foundation for understanding EVC

### **Core Concepts:**
- [ ] Value functions
- [ ] Temporal difference learning
- [ ] Exploration vs. exploitation
- [ ] Learning rates
- [ ] Model-free vs. model-based

### **Resources:**

#### **📚 Books:**

1. **"Reinforcement Learning: An Introduction"** - Sutton & Barto (2nd ed.)
   - **Why:** THE textbook
   - **Time:** 2-3 weeks
   - **Level:** Intermediate
   - **Chapters:** 1-6
   - **Link:** http://incompleteideas.net/book/the-book.html (free)

#### **🎥 Videos:**

1. **David Silver - RL Course**
   - **Link:** https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
   - **Time:** Lectures 1-3 (4.5 hours)
   - **Why:** Best RL course online

2. **DeepMind x UCL - RL Lecture Series**
   - **Link:** https://www.youtube.com/playlist?list=PLqYmG7hTraZBKeNJ-JE_eyJHZ7XgBoAyb
   - **Time:** Lectures 1-2 (3 hours)
   - **Why:** Modern, comprehensive

#### **✍️ Practice:**

```python
# Simple Q-learning example

import numpy as np
import matplotlib.pyplot as plt

class BanditTask:
    """Multi-armed bandit task"""
    def __init__(self, n_arms=4):
        self.n_arms = n_arms
        self.true_values = np.random.randn(n_arms)  # True reward means
    
    def pull(self, arm):
        """Pull an arm, get noisy reward"""
        return self.true_values[arm] + np.random.randn() * 0.5

# Q-learning agent
def q_learning_agent(task, n_trials=1000, learning_rate=0.1, epsilon=0.1):
    """
    Q-learning with epsilon-greedy exploration
    """
    Q = np.zeros(task.n_arms)  # Value estimates
    N = np.zeros(task.n_arms)  # Visit counts
    rewards = []
    
    for trial in range(n_trials):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = np.random.randint(task.n_arms)  # Explore
        else:
            action = np.argmax(Q)  # Exploit
        
        # Get reward
        reward = task.pull(action)
        rewards.append(reward)
        
        # Update Q-value
        Q[action] = Q[action] + learning_rate * (reward - Q[action])
        N[action] += 1
    
    return Q, N, rewards

# Run experiment
task = BanditTask(n_arms=4)
Q, N, rewards = q_learning_agent(task)

print(f"True values: {task.true_values}")
print(f"Learned Q-values: {Q}")
print(f"Visit counts: {N}")

# Plot learning curve
plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(rewards) / np.arange(1, len(rewards)+1))
plt.xlabel('Trial')
plt.ylabel('Average Reward')
plt.title('Learning Curve')
plt.show()
```

---

## 3.3 Hierarchical Gaussian Filter (HGF)

### **Why You Need This:**
- Advanced uncertainty estimation
- Multi-level learning
- Potential upgrade for your project

### **Core Concepts:**
- [ ] Hierarchical belief updating
- [ ] Volatility learning
- [ ] Precision-weighted prediction errors
- [ ] Multi-level uncertainty
- [ ] Generative models

### **Resources:**

#### **📄 Papers (Must Read):**

1. **"A Bayesian foundation for individual learning under uncertainty"** - Mathys et al. (2011)
   - **Why:** Original HGF paper
   - **Time:** 3 hours
   - **Link:** Google Scholar
   - **Note:** Dense but important

2. **"Uncertainty in perception and the HGF"** - Mathys et al. (2014)
   - **Why:** Tutorial paper
   - **Time:** 2 hours
   - **Link:** Google Scholar

#### **🎥 Videos:**

1. **Christoph Mathys - HGF Tutorial**
   - **Search:** "Christoph Mathys HGF" on YouTube
   - **Time:** 1-2 hours
   - **Why:** From the creator

#### **💻 Software:**

1. **TAPAS Toolbox** (MATLAB)
   - **Link:** https://www.tnu.ethz.ch/en/software/tapas
   - **Why:** Official implementation
   - **Time:** 1 day to learn basics

2. **pyhgf** (Python)
   - **Install:** `pip install pyhgf`
   - **Why:** Python implementation
   - **Time:** Half day to learn

#### **✍️ Practice:**

See `HGF_IMPLEMENTATION_GUIDE.md` in your project for implementation examples.

---

# PHASE 4: ADVANCED TOPICS (Week 4-6)

## 4.1 Expected Value of Control (EVC)

### **Why You Need This:**
- This is YOUR framework!
- Foundation of your project
- What you're extending

### **Core Concepts:**
- [ ] Cost-benefit optimization
- [ ] Control as a resource
- [ ] Expected value computation
- [ ] Effort costs
- [ ] ACC as EVC computer

### **Resources:**

#### **📄 Papers (MUST READ - CRITICAL):**

1. **"The expected value of control"** - Shenhav et al. (2013)
   - **Why:** THE foundational paper
   - **Time:** 4 hours (read multiple times!)
   - **Link:** Google Scholar
   - **Note:** THIS IS YOUR STARTING POINT

2. **"Toward a rational and mechanistic account of mental effort"** - Shenhav et al. (2017)
   - **Why:** Extension and review
   - **Time:** 3 hours
   - **Link:** Google Scholar

3. **"The value of control"** - Kool & Botvinick (2018)
   - **Why:** Empirical evidence
   - **Time:** 2 hours
   - **Link:** Google Scholar

#### **🎥 Videos:**

1. **Amitai Shenhav - EVC Framework**
   - **Search:** "Amitai Shenhav EVC" on YouTube
   - **Time:** 1 hour
   - **Why:** From the creator, essential

2. **Matthew Botvinick - Cognitive Control**
   - **Search:** YouTube
   - **Time:** 1 hour
   - **Why:** Co-creator perspective

#### **✍️ Practice:**

```python
# Implement basic EVC model

import numpy as np
import matplotlib.pyplot as plt

def traditional_evc(reward, accuracy, control, effort_cost_weight=1.0):
    """
    Traditional EVC formula
    
    EVC = Expected_Benefit - Expected_Cost
    """
    expected_benefit = reward * accuracy
    expected_cost = effort_cost_weight * control**2
    
    evc = expected_benefit - expected_cost
    return evc

def optimal_control(reward, accuracy, effort_cost_weight=1.0):
    """
    Find optimal control level
    
    For quadratic costs: control* = (reward × accuracy) / (2 × cost_weight)
    """
    return (reward * accuracy) / (2 * effort_cost_weight)

# Example: How does optimal control change with reward?
rewards = np.linspace(0, 10, 50)
accuracy = 0.8
optimal_controls = [optimal_control(r, accuracy) for r in rewards]

plt.figure(figsize=(10, 4))
plt.plot(rewards, optimal_controls)
plt.xlabel('Reward Magnitude')
plt.ylabel('Optimal Control')
plt.title('Traditional EVC: Control Increases with Reward')
plt.grid(True)
plt.show()

# Now add uncertainty (YOUR EXTENSION!)
def bayesian_evc(reward, accuracy, uncertainty, control, 
                 effort_cost_weight=1.0, uncertainty_weight=0.5):
    """
    YOUR Bayesian EVC formula
    
    EVC = Expected_Benefit + Uncertainty_Benefit - Expected_Cost
    """
    expected_benefit = reward * accuracy
    uncertainty_benefit = uncertainty_weight * uncertainty
    expected_cost = effort_cost_weight * control**2
    
    evc = expected_benefit + uncertainty_benefit - expected_cost
    return evc

# Compare: Traditional vs. Bayesian EVC
uncertainties = np.linspace(0, 1, 50)
reward = 5.0
accuracy = 0.8

traditional_controls = [optimal_control(reward, accuracy) for _ in uncertainties]
bayesian_controls = [(reward * accuracy + 0.5 * u) / (2 * 1.0) for u in uncertainties]

plt.figure(figsize=(10, 4))
plt.plot(uncertainties, traditional_controls, label='Traditional EVC', linestyle='--')
plt.plot(uncertainties, bayesian_controls, label='Bayesian EVC (Your Extension)', linewidth=2)
plt.xlabel('Uncertainty')
plt.ylabel('Optimal Control')
plt.title('Key Difference: Bayesian EVC Responds to Uncertainty!')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 4.2 Bayesian Cognitive Models

### **Why You Need This:**
- Understand optimal inference
- Connect to your Bayesian uncertainty
- Theoretical foundation

### **Core Concepts:**
- [ ] Generative models
- [ ] Posterior inference
- [ ] Model comparison
- [ ] Hierarchical models
- [ ] Approximate inference

### **Resources:**

#### **📚 Books:**

1. **"Bayesian Cognitive Modeling"** - Lee & Wagenmakers
   - **Why:** Practical guide
   - **Time:** 2 weeks
   - **Level:** Intermediate
   - **Chapters:** 1-5, 8-10

2. **"Computational Models of Cognition"** - Farrell & Lewandowsky
   - **Why:** Comprehensive overview
   - **Time:** 2 weeks
   - **Level:** Advanced
   - **Chapters:** 7-9

#### **📄 Papers:**

1. **"Bayesian models of cognition"** - Griffiths et al. (2008)
   - **Why:** Overview of Bayesian approach
   - **Time:** 2 hours
   - **Link:** Google Scholar

2. **"Rational use of cognitive resources"** - Lieder & Griffiths (2020)
   - **Why:** Resource-rational framework
   - **Time:** 2 hours
   - **Link:** Google Scholar

---

## 4.3 Hierarchical Bayesian Modeling

### **Why You Need This:**
- Essential for small sample sizes
- Population + individual inference
- Your recommended next step!

### **Core Concepts:**
- [ ] Partial pooling
- [ ] Shrinkage
- [ ] Population vs. individual parameters
- [ ] MCMC sampling
- [ ] Model diagnostics

### **Resources:**

#### **📚 Books:**

1. **"Statistical Rethinking"** - Richard McElreath
   - **Why:** BEST book for Bayesian modeling
   - **Time:** 3-4 weeks
   - **Level:** Intermediate
   - **Chapters:** 1-9, 13
   - **Note:** Highly recommended!

2. **"Bayesian Data Analysis"** - Gelman et al. (3rd ed.)
   - **Why:** The bible of Bayesian stats
   - **Time:** 4-6 weeks
   - **Level:** Advanced
   - **Chapters:** 1-5, 15

#### **🎥 Videos:**

1. **Richard McElreath - Statistical Rethinking Course**
   - **Link:** https://www.youtube.com/playlist?list=PLDcUM9US4XdMROZ57-OIRtIK0aOynbgZN
   - **Time:** 20 lectures × 1.5 hours = 30 hours
   - **Why:** BEST course on Bayesian modeling
   - **Note:** Worth every minute!

#### **💻 Software:**

1. **PyMC Tutorial**
   - **Link:** https://www.pymc.io/projects/docs/en/stable/learn.html
   - **Time:** 1 week
   - **Why:** Tool you'll use

#### **✍️ Practice:**

See `HIERARCHICAL_BAYES_GUIDE.md` in your project for full implementation.

---

# PHASE 5: PROJECT-SPECIFIC (Week 6-8)

## 5.1 Your Bayesian EVC Extension

### **What You're Doing:**

```
Traditional EVC:
EVC = Reward × Accuracy - Cost(Control)

Your Bayesian EVC:
EVC = Reward × Accuracy + λ × Uncertainty - Cost(Control)
                          ↑
                    YOUR CONTRIBUTION!
```

### **Study Your Own Documentation:**

1. **Read in this order:**
   - [ ] `WHY_UNCERTAINTY_MATTERS.md` (motivation)
   - [ ] `UNCERTAINTY_IN_COGNITIVE_MODELS.md` (gap you're filling)
   - [ ] `CONTROL_VARIABLE_EXPLAINED.md` (what you're predicting)
   - [ ] `RUN_STEPS.md` (how to run your code)
   - [ ] `INTERPRETING_RESULTS.md` (what your results mean)

2. **Understand your code:**
   - [ ] `models/traditional_evc.py`
   - [ ] `models/bayesian_evc.py`
   - [ ] `models/bayesian_uncertainty.py`
   - [ ] `utils/data_generator.py`

3. **Run the pipeline:**
   ```bash
   python step1_generate_data.py
   python step2_estimate_uncertainty.py
   python step3_fit_traditional_evc.py
   python step4_fit_bayesian_evc.py
   python step5_compare_models.py
   python step6_visualize.py
   ```

---

## 5.2 Model Fitting & Evaluation

### **Core Concepts:**
- [ ] Maximum likelihood estimation
- [ ] Model comparison (R², AIC, BIC)
- [ ] Cross-validation
- [ ] Parameter recovery
- [ ] Generalization

### **Resources:**

#### **📄 Papers:**

1. **"Model selection and multimodel inference"** - Burnham & Anderson
   - **Why:** Model comparison methods
   - **Time:** 2 hours
   - **Link:** Google Scholar

2. **"Practical Bayesian model evaluation"** - Gelman et al. (2013)
   - **Why:** How to evaluate Bayesian models
   - **Time:** 1.5 hours
   - **Link:** Google Scholar

#### **✍️ Practice:**

```python
# Model comparison example

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Simulate data
true_control = np.random.rand(100)

# Model 1: Simple (no uncertainty)
pred_simple = 0.5 * np.ones(100)

# Model 2: With uncertainty
uncertainty = np.random.rand(100)
pred_uncertainty = 0.3 + 0.4 * uncertainty

# Compare
r2_simple = r2_score(true_control, pred_simple)
r2_uncertainty = r2_score(true_control, pred_uncertainty)

rmse_simple = np.sqrt(mean_squared_error(true_control, pred_simple))
rmse_uncertainty = np.sqrt(mean_squared_error(true_control, pred_uncertainty))

print(f"Simple model: R² = {r2_simple:.3f}, RMSE = {rmse_simple:.3f}")
print(f"Uncertainty model: R² = {r2_uncertainty:.3f}, RMSE = {rmse_uncertainty:.3f}")

# AIC comparison
def aic(log_likelihood, n_params):
    return 2 * n_params - 2 * log_likelihood

# (Compute log likelihoods and compare)
```

---

## 5.3 Computational Psychiatry

### **Why You Need This:**
- Clinical applications
- Individual differences
- Future directions

### **Core Concepts:**
- [ ] Computational phenotyping
- [ ] Parameter-symptom relationships
- [ ] Model-based diagnosis
- [ ] Personalized interventions

### **Resources:**

#### **📄 Papers:**

1. **"Computational psychiatry"** - Huys et al. (2016)
   - **Why:** Overview of field
   - **Time:** 2 hours
   - **Link:** Google Scholar

2. **"Computational phenotyping in psychiatry"** - Browning et al. (2020)
   - **Why:** Clinical applications
   - **Time:** 1.5 hours
   - **Link:** Google Scholar

#### **🎥 Videos:**

1. **Quentin Huys - Computational Psychiatry**
   - **Search:** YouTube
   - **Time:** 1 hour
   - **Why:** Leading researcher

---

# RECOMMENDED LEARNING SCHEDULE

## Week 1: Foundations
- **Mon-Tue:** Probability & Statistics (Seeing Theory, StatQuest)
- **Wed-Thu:** Bayesian Inference Basics (Think Bayes, practice)
- **Fri-Sun:** Cognitive Psychology (Thinking Fast and Slow, videos)

## Week 2: Core Concepts
- **Mon-Tue:** Cognitive Control (papers, videos)
- **Wed-Thu:** Decision-Making Under Uncertainty (papers)
- **Fri-Sun:** Uncertainty in Cognition (papers, practice)

## Week 3: Computational Models Part 1
- **Mon-Tue:** Drift Diffusion Model (papers, simulation)
- **Wed-Thu:** Reinforcement Learning Basics (Sutton & Barto Ch 1-3)
- **Fri-Sun:** RL continued (videos, practice)

## Week 4: Computational Models Part 2
- **Mon-Tue:** HGF (papers, implementation)
- **Wed-Thu:** HGF practice (TAPAS or pyhgf)
- **Fri-Sun:** Review and integrate

## Week 5: Advanced Topics Part 1
- **Mon-Tue:** **EVC Framework (CRITICAL - Shenhav 2013)**
- **Wed-Thu:** EVC continued (Shenhav 2017, practice)
- **Fri-Sun:** Bayesian Cognitive Models (papers)

## Week 6: Advanced Topics Part 2
- **Mon-Wed:** Statistical Rethinking (McElreath, Chapters 1-5)
- **Thu-Fri:** Hierarchical Bayesian Modeling
- **Sat-Sun:** PyMC tutorial and practice

## Week 7: Your Project
- **Mon-Tue:** Read all your project documentation
- **Wed-Thu:** Understand your code thoroughly
- **Fri-Sun:** Run pipeline, experiment with parameters

## Week 8: Integration & Practice
- **Mon-Tue:** Model fitting and evaluation
- **Wed-Thu:** Computational psychiatry applications
- **Fri-Sun:** Write summary, identify gaps, plan next steps

---

# ESSENTIAL SKILLS TO DEVELOP

## Python Programming

### **Core Libraries:**
```bash
pip install numpy scipy pandas matplotlib seaborn
pip install scikit-learn pymc arviz
pip install jupyter notebook
```

### **Practice:**
- [ ] NumPy array operations
- [ ] Pandas data manipulation
- [ ] Matplotlib visualization
- [ ] SciPy optimization
- [ ] PyMC Bayesian modeling

---

## Mathematical Skills

### **Essential Math:**
- [ ] Calculus: Derivatives (for optimization)
- [ ] Linear algebra: Vectors, matrices
- [ ] Probability: Distributions, expectations
- [ ] Optimization: Gradient descent, maximum likelihood

### **Resources:**
1. **3Blue1Brown - Essence of Calculus**
   - **Link:** https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr
   - **Time:** 3 hours

2. **Khan Academy - Linear Algebra**
   - **Link:** https://www.khanacademy.org/math/linear-algebra
   - **Time:** 1 week

---

# KEY PAPERS SUMMARY

## Must Read (In Order):

1. ✅ **Shenhav et al. (2013)** - "The expected value of control"
   - **Why:** YOUR foundation
   - **Priority:** HIGHEST

2. ✅ **Mathys et al. (2011)** - "A Bayesian foundation for individual learning"
   - **Why:** HGF framework
   - **Priority:** HIGH

3. ✅ **Ratcliff & McKoon (2008)** - "The diffusion decision model"
   - **Why:** Decision-making model
   - **Priority:** HIGH

4. ✅ **Behrens et al. (2007)** - "Learning the value of information"
   - **Why:** Uncertainty and learning
   - **Priority:** MEDIUM

5. ✅ **Grupe & Nitschke (2013)** - "Uncertainty and anticipation in anxiety"
   - **Why:** Clinical relevance
   - **Priority:** MEDIUM

---

# QUICK REFERENCE CHEAT SHEET

## Key Formulas:

### **Bayes' Theorem:**
```
P(H|D) = P(D|H) × P(H) / P(D)

Posterior = Likelihood × Prior / Evidence
```

### **Traditional EVC:**
```
EVC = Reward × Accuracy - Cost(Control)

Optimal Control = (Reward × Accuracy) / (2 × Cost_Weight)
```

### **Your Bayesian EVC:**
```
EVC = Reward × Accuracy + λ × Uncertainty - Cost(Control)

Optimal Control = (Reward × Accuracy + λ × Uncertainty) / (2 × Cost_Weight)
```

### **DDM:**
```
dx/dt = v + σ × ε(t)

v = drift rate (evidence strength)
σ = diffusion noise (uncertainty)
```

### **Kalman Filter:**
```
K = σ²_pred / (σ²_pred + σ²_obs)  # Kalman gain

μ_new = μ_pred + K × (observation - μ_pred)
```

### **HGF:**
```
Learning Rate = f(uncertainty, volatility)

Higher uncertainty → Higher learning rate
```

---

# ASSESSMENT CHECKLIST

After completing this roadmap, you should be able to:

## Conceptual Understanding:
- [ ] Explain what cognitive control is and why it matters
- [ ] Describe how uncertainty affects decision-making
- [ ] Articulate the gap in traditional EVC
- [ ] Explain your Bayesian EVC extension
- [ ] Connect to clinical applications

## Technical Skills:
- [ ] Implement basic Bayesian inference
- [ ] Fit computational models to data
- [ ] Evaluate model performance (R², RMSE, etc.)
- [ ] Use PyMC for hierarchical modeling
- [ ] Visualize results effectively

## Project-Specific:
- [ ] Run your entire pipeline
- [ ] Interpret your results
- [ ] Explain λ (uncertainty weight) parameter
- [ ] Discuss limitations and future directions
- [ ] Present your work clearly

---

# FINAL TIPS

## 1. **Don't Rush**
- Take time to understand fundamentals
- It's okay to spend extra time on difficult concepts
- Revisit topics as needed

## 2. **Practice, Practice, Practice**
- Code along with tutorials
- Implement models from scratch
- Experiment with parameters

## 3. **Connect to Your Project**
- Always ask: "How does this relate to my work?"
- Try to implement concepts in your codebase
- Test ideas with your data

## 4. **Use Multiple Resources**
- Papers for depth
- Videos for intuition
- Code for practice
- Combine all three!

## 5. **Ask Questions**
- Join online communities (r/statistics, r/MachineLearning)
- Use Stack Overflow for coding issues
- Discuss with colleagues

## 6. **Document Your Learning**
- Keep a learning journal
- Write summaries of papers
- Create your own examples

---

# NEXT STEPS AFTER THIS ROADMAP

## Immediate (Week 9-10):
1. Implement hierarchical Bayesian EVC
2. Test on real data (OpenNeuro datasets)
3. Write up results

## Short-term (Month 3-4):
1. Submit to conference (Cognitive Science Society, CCN)
2. Preprint on bioRxiv
3. Collect pilot data

## Long-term (Month 6-12):
1. Full manuscript for journal
2. Clinical validation study
3. Extend to other domains

---

# RESOURCES SUMMARY

## Free Online Books:
- Think Bayes: https://greenteapress.com/wp/think-bayes/
- Sutton & Barto RL: http://incompleteideas.net/book/
- Statistical Rethinking lectures: https://github.com/rmcelreath/stat_rethinking_2023

## Video Courses:
- 3Blue1Brown: https://www.youtube.com/c/3blue1brown
- StatQuest: https://www.youtube.com/c/joshstarmer
- Richard McElreath: https://www.youtube.com/c/rmcelreath

## Software:
- PyMC: https://www.pymc.io/
- ArviZ: https://arviz-devs.github.io/arviz/
- TAPAS: https://www.tnu.ethz.ch/en/software/tapas

## Communities:
- r/statistics: https://www.reddit.com/r/statistics/
- r/MachineLearning: https://www.reddit.com/r/MachineLearning/
- PyMC Discourse: https://discourse.pymc.io/

---

**Good luck with your learning journey! 🚀**

**Remember:** Understanding takes time. Be patient with yourself, practice consistently, and always connect concepts back to your project. You're doing important work!

