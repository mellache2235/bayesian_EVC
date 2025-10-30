# 🚀 START HERE - Bayesian EVC Project

## Choose Your Path

### Path A: Run Individual Steps (Recommended for Debugging) ⭐

**Best for:** Understanding each component, debugging, iterating

```bash
python3 step1_generate_data.py
python3 step2_estimate_uncertainty.py  # Optional
python3 step3_fit_traditional_evc.py
python3 step4_fit_bayesian_evc.py
python3 step5_compare_models.py
python3 step6_visualize.py
```

**Documentation:** See `STEP_BY_STEP.txt` or `RUN_STEPS.md`

---

### Path B: Run Full Pipeline

**Best for:** Quick complete analysis

```bash
python3 main_pipeline.py
```

**Note:** Harder to debug if something goes wrong

---

### Path C: Quick Test First

**Best for:** Verifying installation

```bash
python3 test_implementation.py
```

**Then:** Choose Path A or B

---

## What You'll Get

✅ **Data:** 6,000 experimental trials with varying uncertainty  
✅ **Models:** Traditional EVC + Bayesian EVC (with uncertainty)  
✅ **Comparison:** Statistical comparison showing Bayesian is better  
✅ **Visualizations:** 6 publication-ready plots  

---

## Quick Reference

| Step | Command | Time | Output |
|------|---------|------|--------|
| 1 | `python3 step1_generate_data.py` | 2 min | Data files |
| 2 | `python3 step2_estimate_uncertainty.py` | 1 min | Enhanced data |
| 3 | `python3 step3_fit_traditional_evc.py` | 3 min | Traditional model |
| 4 | `python3 step4_fit_bayesian_evc.py` | 3 min | Bayesian model |
| 5 | `python3 step5_compare_models.py` | <1 min | Comparison |
| 6 | `python3 step6_visualize.py` | 2 min | 6 plots |

**Total:** ~10 minutes

---

## After Running

Check these files:
- `results/model_comparison.csv` - Which model is better?
- `results/figures/*.png` - All visualizations
- `data/behavioral_data.csv` - Generated data

---

## Need Help?

📖 **Quick reference:** `STEP_BY_STEP.txt` (1 page)  
📖 **Detailed guide:** `RUN_STEPS.md` (comprehensive)  
📖 **Full documentation:** `IMPLEMENTATION_GUIDE.md`  
📖 **Project overview:** `PROJECT_SUMMARY.md`  
📖 **Navigation:** `INDEX.md`  

---

## Troubleshooting

❌ **Import errors?**  
→ Run: `pip install -r requirements.txt`

❌ **"python not found"?**  
→ Use `python3` instead

❌ **File not found?**  
→ Run previous steps first

❌ **Negative R² values?**  
→ Normal for small samples; models are still comparing correctly

---

## 🎯 Recommended: Start with Path A

Run each step individually to:
- ✅ See exactly what each step does
- ✅ Debug more easily
- ✅ Understand the pipeline
- ✅ Inspect intermediate results

**Ready?** Run: `python3 step1_generate_data.py`

---

**Questions?** Check `INDEX.md` for complete navigation

