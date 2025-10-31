"""
Generate Arithmetic Task Data for Bayesian EVC Analysis

Simulates children (ages 7-12) solving arithmetic problems of varying difficulty.
Incorporates:
- Learning from past trials (HGF-like belief updating)
- Uncertainty from problem difficulty
- Individual differences in ability and control allocation
- Realistic performance (RT, accuracy)

Output: data/arithmetic/arithmetic_task_data.csv
"""

import numpy as np
import pandas as pd
import os


class ArithmeticTaskGenerator:
    """Generate arithmetic task data for children"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
    
    def generate_arithmetic_problem(self, difficulty):
        """
        Generate arithmetic problem of given difficulty
        
        Args:
            difficulty: 1 (easy) to 5 (very hard)
            
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
            # Medium: two-digit with subtraction possible
            a = np.random.randint(20, 100)
            b = np.random.randint(10, 50)
            operation = np.random.choice(['+', '-'])
            if operation == '-' and b > a:
                a, b = b, a
                
        elif difficulty == 4:
            # Hard: three-digit or multiplication
            if np.random.rand() > 0.5:
                a = np.random.randint(100, 500)
                b = np.random.randint(100, 500)
                operation = '+'
            else:
                a = np.random.randint(10, 30)
                b = np.random.randint(10, 30)
                operation = '*'
                
        else:  # difficulty == 5
            # Very hard: complex operations
            a = np.random.randint(100, 999)
            b = np.random.randint(100, 999)
            operation = np.random.choice(['+', '-', '*'])
            if operation == '-' and b > a:
                a, b = b, a
            if operation == '*':
                a = a // 100
                b = b // 100
        
        # Compute answer
        if operation == '+':
            answer = a + b
        elif operation == '-':
            answer = a - b
        else:
            answer = a * b
        
        problem_string = f"{a} {operation} {b}"
        
        return {
            'problem': problem_string,
            'operand1': a,
            'operand2': b,
            'operation': operation,
            'correct_answer': answer,
            'difficulty': difficulty
        }
    
    def generate_child_data(self, child_id, n_trials=100, 
                           math_ability=0.7, uncertainty_tolerance=0.5, 
                           persistence=0.7):
        """Generate data for one child"""
        
        trials = []
        
        # Initial beliefs (like HGF state)
        belief_about_ability = math_ability
        uncertainty_about_ability = 0.5
        
        for trial in range(n_trials):
            # Select difficulty (adaptive based on performance)
            if trial < 10:
                difficulty = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            else:
                if belief_about_ability > 0.7:
                    difficulty = np.random.choice([2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.2])
                elif belief_about_ability > 0.5:
                    difficulty = np.random.choice([1, 2, 3, 4], p=[0.2, 0.3, 0.3, 0.2])
                else:
                    difficulty = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            
            # Generate problem
            problem = self.generate_arithmetic_problem(difficulty)
            
            # Compute variables for control allocation
            reward = difficulty * 10  # Harder = more points
            base_accuracy = math_ability * (1.1 - 0.2 * difficulty)
            base_accuracy = np.clip(base_accuracy, 0.1, 0.95)
            
            # Uncertainty
            problem_uncertainty = (difficulty - 1) / 4
            state_uncertainty = uncertainty_about_ability
            total_uncertainty = 0.6 * problem_uncertainty + 0.4 * state_uncertainty
            
            # Bayesian EVC: Control allocation
            baseline_control = 0.3
            expected_value = reward * base_accuracy
            evc_component = expected_value * 0.1
            uncertainty_component = uncertainty_tolerance * total_uncertainty * 0.4
            difficulty_boost = persistence * (difficulty / 5) * 0.3
            
            control_signal = baseline_control + evc_component + \
                           uncertainty_component + difficulty_boost + \
                           np.random.normal(0, 0.1)
            control_signal = np.clip(control_signal, 0, 1)
            
            # Simulate performance
            actual_accuracy = base_accuracy + control_signal * 0.2
            actual_accuracy = np.clip(actual_accuracy, 0.05, 0.98)
            correct = np.random.rand() < actual_accuracy
            
            # Reaction time
            base_rt = 1000 + difficulty * 500
            control_rt_increase = control_signal * 1000
            rt = base_rt + control_rt_increase + np.random.normal(0, 200)
            rt = max(rt, 300)
            
            # Confidence
            confidence = 1 - total_uncertainty
            confidence = np.clip(confidence, 0.1, 0.9)
            
            # Update beliefs (HGF-like)
            prediction_error = (1 if correct else 0) - belief_about_ability
            learning_rate = 0.1 + 0.1 * uncertainty_about_ability
            belief_about_ability += learning_rate * prediction_error
            belief_about_ability = np.clip(belief_about_ability, 0.1, 0.9)
            
            if correct:
                uncertainty_about_ability *= 0.95
            else:
                uncertainty_about_ability = min(uncertainty_about_ability * 1.05, 0.8)
            
            # Store trial
            trials.append({
                'child_id': child_id,
                'trial': trial + 1,
                'problem': problem['problem'],
                'difficulty': difficulty,
                'correct_answer': problem['correct_answer'],
                'math_ability': math_ability,
                'uncertainty_tolerance': uncertainty_tolerance,
                'persistence': persistence,
                'problem_uncertainty': problem_uncertainty,
                'state_uncertainty': state_uncertainty,
                'total_uncertainty': total_uncertainty,
                'belief_about_ability': belief_about_ability,
                'control_signal': control_signal,
                'correct': int(correct),
                'rt': rt,
                'confidence': confidence,
                'reward': reward,
                'expected_accuracy': base_accuracy
            })
        
        return pd.DataFrame(trials)
    
    def generate_dataset(self, n_children=30, n_trials_per_child=100, 
                        age_range=(7, 12)):
        """Generate full dataset"""
        
        all_data = []
        
        print(f"Generating arithmetic task data for {n_children} children...")
        print(f"  Age range: {age_range[0]}-{age_range[1]} years")
        print(f"  Trials per child: {n_trials_per_child}")
        
        for child_id in range(n_children):
            # Child characteristics
            age = np.random.uniform(age_range[0], age_range[1])
            base_ability = 0.3 + (age - age_range[0]) / (age_range[1] - age_range[0]) * 0.4
            math_ability = base_ability + np.random.normal(0, 0.15)
            math_ability = np.clip(math_ability, 0.2, 0.9)
            
            uncertainty_tolerance = np.random.uniform(0.2, 0.8)
            persistence = np.random.uniform(0.4, 0.9)
            
            # Generate data
            child_data = self.generate_child_data(
                child_id=child_id,
                n_trials=n_trials_per_child,
                math_ability=math_ability,
                uncertainty_tolerance=uncertainty_tolerance,
                persistence=persistence
            )
            
            child_data['age'] = age
            all_data.append(child_data)
            
            if (child_id + 1) % 10 == 0:
                print(f"  Generated {child_id + 1}/{n_children} children...")
        
        dataset = pd.concat(all_data, ignore_index=True)
        
        print(f"\n✓ Generated {len(dataset)} trials total")
        print(f"  Mean accuracy: {dataset['correct'].mean():.2%}")
        
        return dataset


def main():
    print("=" * 70)
    print("GENERATE ARITHMETIC TASK DATA")
    print("=" * 70)
    
    # Generate data
    generator = ArithmeticTaskGenerator(seed=42)
    
    data = generator.generate_dataset(
        n_children=30,
        n_trials_per_child=100,
        age_range=(7, 12)
    )
    
    # Save
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
        if len(diff_data) > 0:
            print(f"  Level {diff}: N={len(diff_data):4d}, "
                  f"Accuracy={diff_data['correct'].mean():.2%}, "
                  f"RT={diff_data['rt'].mean():4.0f}ms")
    
    print(f"\nBy age group:")
    for age_group in [(7, 8), (8, 9), (9, 10), (10, 11), (11, 12)]:
        age_data = data[(data['age'] >= age_group[0]) & (data['age'] < age_group[1])]
        if len(age_data) > 0:
            n_children = age_data['child_id'].nunique()
            print(f"  Age {age_group[0]}-{age_group[1]}: N={n_children} children, "
                  f"Accuracy={age_data['correct'].mean():.2%}")
    
    print("\n" + "=" * 70)
    print("✓ ARITHMETIC DATA GENERATION COMPLETE!")
    print("=" * 70)
    print("\nNext step: Run 'python3 fit_bayesian_evc_arithmetic.py'")


if __name__ == '__main__':
    main()

