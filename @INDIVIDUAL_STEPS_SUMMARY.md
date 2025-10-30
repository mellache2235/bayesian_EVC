# Bayesian EVC Pipeline — Individual Steps Summary

1. **Set up environment**  
   - Ensure Python 3.10+ is available.  
   - Install project requirements (`pip install -r requirements.txt`). If `pandas` or `h5py` wheels fail due to old compilers, pin to compatible versions or install via conda as noted in the main documentation.

2. **Generate structured synthetic data**  
   - Run the CLI: `python3 scripts/generate_data.py`.  
   - Optional flags: `--children <N>` `--trials <N>` `--seed <int>` `--output <path>` for customised datasets.  
   - Output: `data/structured_evc_trials.csv` (defaults) containing realistic EF task dynamics (volatility, clarity, control signals).  
   - Reference: `src/data_generation.py` for underlying assumptions and parameter sampling.

3. **Inspect dataset (optional)**  
   - Use a notebook or `pandas` shell snippet to view distributions:  
     ```python
     import pandas as pd
     df = pd.read_csv("data/structured_evc_trials.csv")
     df.describe()
     ```

4. **Run the Bayesian EVC pipeline**  
   - Execute `python3 -m src.pipeline` (defaults to the structured dataset).  
   - The script prints trial-level estimates (posterior uncertainty, predicted control) and per-child summaries.

5. **Compare to non-Bayesian baseline (optional)**  
   - Modify `EVCWeights.uncertainty_reduction` in `src/pipeline.py` (e.g., set to 0) to emulate a traditional reward–effort model.  
   - Rerun the pipeline and contrast control allocations or accuracy predictions against the Bayesian variant.

6. **Extend / customise**  
   - Adjust hyperparameters in `GenerationConfig` or `ModelConfig` for alternative task structures.  
   - Add visualisations or model-fitting scripts as needed (e.g., integrate PyMC/NumPyro for richer inference).  
   - Document new experiments in `docs/` to keep assumptions transparent.

7. **Regenerate data when parameters change**  
   - Re-run `scripts/generate_data.py` whenever you tweak the generator; commit the updated CSV if reproducibility is required.

8. **Clean up artefacts (optional)**  
   - Delete temporary datasets (e.g., `data/test_structured.csv`) once finished evaluating.

This file tracks the hands-on workflow so you can iterate quickly without re-reading the full proposal.

