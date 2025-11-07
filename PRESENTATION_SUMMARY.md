# Presentation Summary & Talking Points

**File:** `presentation/UQ_Capstone_Presentation.pptx`  
**Created:** November 7, 2025  
**Total Slides:** 20

## Presentation Outline

### SLIDES 1-4: Introduction & Setup

**Slide 1: Title Slide**
- Title: "Uncertainty Quantification for Medical Image Classification"
- Subtitle: "Capstone Project - Rutgers University, November 2025"

**Slide 2: Problem Statement**
- Why UQ matters: Model confidence ≠ correctness
- Real-world example: 92% confidence doesn't mean 92% accuracy
- Key question: When can we trust model predictions?
- Solution: Compare 4 UQ methods on same dataset

**Slide 3: Dataset Overview**
- Chest X-Ray binary classification (Normal vs Pneumonia)
- Split: 4,172 training, 1,044 calibration, 624 test
- Model: ResNet-18 (pretrained)
- Baseline accuracy: 91.67%

**Slide 4: UQ Methods Overview**
- **Baseline:** Standard ResNet-18 (point estimate)
- **MC Dropout:** 15 forward passes with dropout enabled
- **Deep Ensemble:** 5 independently trained models
- **SWAG:** Bayesian posterior approximation (30 samples)

---

### SLIDES 5-6: Core Results

**Slide 5: Accuracy Comparison (Visual)**
- Includes embedding of comprehensive_metrics_visualization.png
- 9-panel dashboard with all key metrics

**Slide 6: Results Summary Table**

| Method | Accuracy | ECE | Brier | FNR |
|--------|----------|-----|-------|-----|
| Baseline | 91.67% | 0.0498 | 0.0704 | 0.0833 |
| MC Dropout | 85.26% | 0.1172 | 0.1246 | 0.1474 |
| Deep Ensemble | 91.67% | **0.0271** | 0.0630 | 0.0833 |
| SWAG | 83.17% | 0.1519 | 0.1528 | 0.1683 |

---

### SLIDES 7-10: Analysis & Insights

**Slide 7: MC Dropout Success Story**
- Initial problem: 66% accuracy (expected ~90%)
- Root cause: Dropout enabled once, accumulated across all 15 samples
- Solution: Toggle dropout on/off between each forward pass
- Result: 66% → 85.26% (FIXED!)
- **Key lesson:** Implementation details matter critically in UQ

**Slide 8: SWAG Underperformance**
- Expected: ~91% (competitive with Ensemble)
- Actual: 83.17% (underperformer)
- Root cause: Validation overfitting (99.62% val vs 85.58% test)
- Why: Snapshot collection caught overfit region
- Insight: Bayesian posterior sampling includes uncertainty itself
- When posterior concentrated on overfit → poor out-of-distribution

**Slide 9: Winner - Deep Ensemble**
- Accuracy: 91.67%
- **Best calibration: ECE = 0.0271**
- Brier score: 0.0630
- Why it dominates:
  - Model diversity from 5 independent initializations
  - Disagreement quantifies epistemic uncertainty
  - Simple, interpretable, robust
- Trade-off: 5x computational cost

**Slide 10: Method Comparison - Pros & Cons**
- **MC Dropout:** Fast, minimal memory, but lower accuracy & weak calibration
- **Ensemble:** Best overall, but 5x inference cost
- **SWAG:** Theoretically sound, but sensitive to hyperparameters
- **Recommendation:** Ensemble if budget allows; MC Dropout if speed critical

---

### SLIDES 11-12: Advanced Topics

**Slide 11: Conformal Risk Control**
- What: Distribution-free confidence guarantees
- Goal: Prediction sets with specified coverage
- Methods implemented:
  - **FNR Control:** Minimizes false negatives (critical for medical)
  - **Set Size Control:** Minimizes prediction set size
  - **Composite Loss:** Balances coverage & size
- Why important: Formal guarantees robust to domain shift

**Slide 12: Segmentation vs Classification Context**
- Pattern across tasks: **Ensemble consistently strong**
- MC Dropout shows potential but needs tuning
- SWAG struggles with regularization
- Cross-task lesson: Diversity mitigates regularization issues

---

### SLIDES 13-14: Lessons & Future Work

**Slide 13: Key Lessons**
1. Implementation details matter (dropout toggle)
2. Overfitting affects UQ methods differently
3. Calibration ≠ Accuracy (both matter for medical AI)
4. Conformal methods provide formal guarantees

**Slide 14: Next Steps & Future Directions**
- Short-term: Finalize conformal prediction, deployment comparisons
- Medium-term: Improve SWAG regularization, test on OOD data
- Research: Ensemble + Conformal for certified safety, task-aware UQ

---

### SLIDES 15-19: Research Collaboration with Dr. Gemma Moran

**Slide 15: Research Collaboration Overview**
- Your background: Biomathematics + Data Science
- Dr. Moran's expertise: Statistical Learning Theory
- Goal: Practical ML + Statistical rigor

**Slide 16: Main Idea - Adaptive Bayesian Ensembles**
- Core insight: Not all uncertainty equally important
- Medical imaging: Epistemic >> Aleatoric
- Method: Learn task-specific weights for UQ components
- Innovation: Combines multiple UQ paradigms adaptively
- **Why novel:** Not yet published; bridges biomathematics + ML
- Roles: You (framework + validation), Dr. Moran (theory + guarantees)

**Slide 17: Alternative Collaboration Ideas**
- **Option 2:** Conformal regression for risk prediction
- **Option 3:** UQ quality metrics framework
- **Option 4:** Interpretable UQ for clinicians
- Common theme: Practical, not overly theoretical, medical focus

**Slide 18: Timeline & Collaboration Structure**
- **Phase 1 (Dec):** Finalize capstone, literature review, initial experiments
- **Phase 2 (Jan-Mar):** Develop theory, implement, comprehensive comparison
- **Phase 3 (Apr-May):** Write manuscript, submit to conference (MICCAI, ISBI)
- **Deliverables:** 1-2 first-author papers, open-source code

**Slide 19: Your Unique Position**
- Why this collaboration works:
  - You bring applied perspective to statistical theory
  - Dr. Moran brings rigor to practical methods
  - Perfect modern ML research match
- Current moment: UQ in medical AI is hot topic
- Your capstone shows genuine research potential

---

### SLIDE 20: Closing
- Questions?
- Thank you
- Contact + GitHub

---

## Key Talking Points by Section

### For Results Section (Slides 5-10)

**What to emphasize:**
1. "Deep Ensemble is the clear winner with 91.67% accuracy AND the best calibration at ECE=0.0271"
2. "MC Dropout's success story: We went from 66% to 85.26% by fixing a subtle dropout toggle bug"
3. "SWAG's issue is validation overfitting - it's not the method itself, but regularization during training"
4. "Across both segmentation and classification, Ensemble consistently outperforms"
5. "MC Dropout stays in the middle - good for computational constraints"

### For SWAG Analysis (Slide 8)

**Explain the 83% issue:**
- "SWAG achieved 99.62% on validation but only 85% on the test set during training"
- "This massive gap indicates validation overfitting"
- "When we sampled from the Bayesian posterior, we were sampling from an overfit distribution"
- "It's like your posterior learned the training noise instead of the signal"
- "Solution: Better regularization during snapshot collection OR conservative snapshot timing"

### For Conformal Methods (Slide 11)

**Why it matters:**
- "We can now make predictions with formal coverage guarantees"
- "No matter what the hospital's imaging protocol is, our guarantee holds"
- "For medical AI, this is huge: We can promise clinicians 'this prediction set contains truth 95% of the time'"

### For Research Collaboration (Slides 15-19)

**Your pitch:**
- "This capstone project is just the beginning"
- "There's a really interesting research direction here combining Bayesian + Ensemble methods adaptively"
- "We could have something publication-ready by May"
- "This combines your theoretical interests with practical medical AI applications"

---

## Before Your Presentation

### Practice Tips:
1. **Tell the MC Dropout story well** - It's your best narrative
2. **Be honest about SWAG** - Show you understand the limitation
3. **Emphasize ensemble's reliability** - Boring but important
4. **End strong** - Make the research collaboration sound exciting

### Optional Additions:
- Show one visualization live (e.g., comprehensive_metrics_visualization.png)
- Have backup: "Here's what SWAG could achieve with better regularization..."
- Have a question ready: "Which UQ method would YOU use in a production medical system?"

### Time Allocation:
- 0-2 min: Problem statement & motivation
- 2-5 min: Methods overview
- 5-10 min: Results & analysis (this is where you shine)
- 10-12 min: MC Dropout debugging story (audience favorite)
- 12-15 min: SWAG analysis and ensemble superiority
- 15-18 min: Conformal methods & lessons
- 18-22 min: Research collaboration ideas
- 22-25 min: Questions

---

## File Locations

- **Presentation:** `presentation/UQ_Capstone_Presentation.pptx`
- **Generation script:** `create_presentation.py` (can modify and regenerate)
- **Embedded visualizations:** `runs/classification/metrics/comprehensive_metrics_visualization.png`
- **Metrics data:** `runs/classification/metrics/metrics_summary.csv`

---

## Next Steps After Creating Presentation

1. **Review all 20 slides** in PowerPoint
2. **Test embedded images** - verify they display correctly
3. **Customize:** Add your name, email, GitHub link on final slide
4. **Practice delivery** - especially the SWAG & MC Dropout sections
5. **Print speaker notes** if needed
6. **Record video walkthrough** for YouTube (optional)

---

## Research Collaboration Ideas (To Discuss with Dr. Moran)

### Strongest Proposal: Adaptive Bayesian Ensembles
- **Problem:** Medical imaging needs both epistemic AND aleatoric uncertainty
- **Solution:** Learn task-specific importance weights
- **Innovation:** Adaptive combination of Ensemble + Bayesian methods
- **Timeline:** Feasible as paper by May 2025
- **Your background:** Perfect fit (biomathematics + data science)

### Fallback Proposals:
1. Conformal prediction for disease severity scoring
2. UQ quality metrics framework (formal definitions)
3. Interpretable uncertainty for clinicians

All are publication-ready with Dr. Moran's involvement.

---

Good luck with your presentation! 🎓
