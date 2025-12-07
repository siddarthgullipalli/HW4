# 📸 SCREENSHOT PLACEMENT GUIDE FOR HW4 REPORT
## Quick Reference for All Required Screenshots

---

## Total Screenshots Needed: 11

### ✅ PART 1: REWARD MODEL TRAINING (3 screenshots)

#### Screenshot 1: Preprocessed Data Table ⭐ CRITICAL
**Location in Notebook:** Cell output showing table with columns:
- `history`
- `human_ref_A`
- `human_ref_B`
- `preferred`

**What to capture:**
```
====================================================================================================
PREPROCESSED REWARD MODEL TRAINING DATA (First 5 Examples)
====================================================================================================
[TABLE WITH 5 ROWS]
```

**Where to put in report:** Section 2.2.2 "Preprocessed Data Table (REQUIRED)"

**Why it's needed:** Assignment explicitly requires showing preprocessed data format

---

#### Screenshot 2: Training Progress (First 3 Batches)
**Location in Notebook:** Cell output showing:
```
Batch 1: Loss = 0.3705
Batch 2: Loss = 0.0267
Batch 3: Loss = 0.2505
```

**Where to put in report:** Section 2.5 "Training Process"

**Why it's needed:** Show first few epochs of training (assignment requirement)

---

#### Screenshot 3: Reward Model Training Curves
**Location in Notebook:** Image output showing plot

**File saved as:** `reward_model_training.png`

**What it shows:**
- Blue line: Training Loss over 5 epochs
- Orange line: Validation Loss over 5 epochs
- Both decreasing trend

**Where to put in report:** Section 2.5.2 "Training Curve Analysis"

**Why it's needed:** Learning curves are explicitly required in assignment

---

### ✅ PART 2: PPO FINE-TUNING (2 screenshots)

#### Screenshot 4: Full PPO Training Start
**Location in Notebook:** Cell output showing:
```
================================================================================
PPO TRAINING - FULL FINE-TUNING
================================================================================
```

**Where to put in report:** Section 3.4.2 "Training Process"

**Why it's needed:** Show training initialization and progress

---

#### Screenshot 5: LoRA Parameter Efficiency
**Location in Notebook:** Cell output showing:
```
trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.2364
```

**Where to put in report:** Section 3.5.2 "Parameter Efficiency"

**Why it's needed:** Demonstrates LoRA's parameter efficiency advantage

---

### ✅ PART 3: EVALUATION & REPORTING (4 screenshots)

#### Screenshot 6: Reward Score Evaluation Summary
**Location in Notebook:** Cell output showing:
```
================================================================================
REWARD SCORE EVALUATION
================================================================================
Pretrained:  0.0440
Full PPO:    0.0480  (gain: +0.0040)
LoRA PPO:    0.0396  (gain: -0.0044)
================================================================================
```

**Where to put in report:** Section 4.2.2 "Results Summary"

**Why it's needed:** Core quantitative results for evaluation

---

#### Screenshot 7: Reward Distribution Plots
**Location in Notebook:** Image output

**File saved as:** `reward_evaluation.png`

**What it shows:**
- Left: Box plot comparing Pretrained, Full, LoRA
- Right: Bar chart of average rewards

**Where to put in report:** Section 4.2.4 "Reward Distribution"

**Why it's needed:** Visual representation of model comparison

---

#### Screenshot 8: KL Divergence Plots ⭐ CRITICAL
**Location in Notebook:** Image output

**File saved as:** `kl_divergence.png`

**What it shows:**
- Left: Full Fine-tuning KL over training steps
- Right: LoRA Fine-tuning KL over training steps
- Reference lines at 0.02 (min) and 0.1 (max)

**Where to put in report:** Section 4.3 "KL Divergence Monitoring"

**Why it's needed:** Assignment explicitly requires KL plot with interpretation

---

#### Screenshot 9: Manual Evaluation Prompts ⭐ CRITICAL
**Location in Notebook:** Cell output showing:
```
================================================================================
MANUAL EVALUATION PROMPTS
================================================================================

PROMPT 1
[Spanish drinking question]
--- Pretrained ---
[response]
--- Full PPO ---
[response]
--- LoRA PPO ---
[response]

PROMPT 2
[Chinese restaurant question]
[responses from all 3 models]

PROMPT 3
[GDPR synthetic data question]
[responses from all 3 models]
```

**Where to put in report:** Section 4.4.2 "Selected Prompts for Manual Evaluation"

**Why it's needed:** Required for manual rating tables (must rate 3 prompts)

---

### ✅ BONUS: DCGAN (2 screenshots)

#### Screenshot 10: DCGAN Training Progress
**Location in Notebook:** Cell output showing:
```
Epoch 100: D=0.2889, G=1.9693
Epoch 200: D=0.2171, G=2.6002
...
Epoch 1000: D=0.0653, G=4.9040
```

**Where to put in report:** Section 5.5.1 "Training Progression"

**Why it's needed:** Show training progress over 1000 epochs

---

#### Screenshot 11: Generated Medical Images Grid ⭐ CRITICAL
**Location in Notebook:** Image output showing 8×4 grid

**File saved as:** `dcgan_samples.png`

**What it shows:**
- 32 synthetic chest X-ray images (8 columns × 4 rows)
- Grayscale medical images

**Where to put in report:** Section 5.6 "Generated Samples"

**Why it's needed:** Assignment requires displaying minimum 32 synthetic images in 8×4 grid

---

## 📊 IMAGE FILES GENERATED BY NOTEBOOK

1. **reward_model_training.png** - Training curves for reward model
2. **reward_evaluation.png** - Box plot and bar chart of rewards
3. **kl_divergence.png** - KL divergence monitoring plots
4. **dcgan_samples.png** - 8×4 grid of generated medical images

**How to include in PDF report:**
- Insert these PNG files directly into your Word/Google Doc
- Make sure they're high resolution (saved at 300 DPI)
- Add captions below each image

---

## ✍️ TABLES YOU NEED TO CREATE MANUALLY

### Manual Rating Tables (9 total - 3 models × 3 prompts)

For each of the 3 prompts shown in Screenshot 9, you need to create a table rating each model's response on a scale of 1-5:

**Rating Categories:**
1. **Coherence** (1=gibberish, 5=perfect grammar)
2. **Relevance** (1=off-topic, 5=directly answers question)
3. **Helpfulness** (1=useless, 5=very helpful)
4. **Completeness** (1=incomplete, 5=fully answers)
5. **Hallucination/Toxicity** (1=severe issues, 5=none)

**Example Table Format:**

```
Table 1: Manual Ratings - Pretrained Model (Prompt 1)
┌─────────────────────────┬──────────────────┐
│ Generated Response      │ Ratings (1-5)    │
├─────────────────────────┼──────────────────┤
│ [paste response here]   │ Coherence: 2     │
│                         │ Relevance: 1     │
│                         │ Helpfulness: 1   │
│                         │ Completeness: 1  │
│                         │ Hallucination: 3 │
└─────────────────────────┴──────────────────┘
```

**YOU MUST CREATE 9 TABLES:**
- Prompt 1: Pretrained, Full PPO, LoRA PPO (3 tables)
- Prompt 2: Pretrained, Full PPO, LoRA PPO (3 tables)
- Prompt 3: Pretrained, Full PPO, LoRA PPO (3 tables)

---

## 🎯 CRITICAL SCREENSHOTS (Don't Miss These!)

These screenshots are **explicitly required** by the assignment:

1. ✅ **Screenshot 1**: Preprocessed data table (5 rows minimum)
2. ✅ **Screenshot 8**: KL divergence plot with interpretation
3. ✅ **Screenshot 9**: Manual evaluation prompts (for rating)
4. ✅ **Screenshot 11**: DCGAN generated images (8×4 grid)

---

## 📋 SCREENSHOT CHECKLIST

Before submitting, verify you have:

- [ ] Screenshot 1: Preprocessed data table
- [ ] Screenshot 2: First 3 training batches
- [ ] Screenshot 3: Reward training curves (PNG file)
- [ ] Screenshot 4: Full PPO training start
- [ ] Screenshot 5: LoRA parameter efficiency
- [ ] Screenshot 6: Reward evaluation summary
- [ ] Screenshot 7: Reward distribution plots (PNG file)
- [ ] Screenshot 8: KL divergence plots (PNG file)
- [ ] Screenshot 9: Manual evaluation prompts with responses
- [ ] Screenshot 10: DCGAN training progress
- [ ] Screenshot 11: DCGAN generated images grid (PNG file)

**Image files:**
- [ ] reward_model_training.png
- [ ] reward_evaluation.png
- [ ] kl_divergence.png
- [ ] dcgan_samples.png

**Manual tables:**
- [ ] 9 rating tables completed (3 prompts × 3 models)

---

## 💡 TIPS FOR TAKING SCREENSHOTS

### For Code Output Screenshots:
1. **Run the notebook cell**
2. **Expand the output** if truncated
3. **Take screenshot** of the relevant output section
4. **Crop** to show only relevant content
5. **Save** with descriptive filename (e.g., "preprocessed_data.png")

### For Plot Screenshots:
1. Plots are **automatically saved** as PNG files by the notebook
2. Find them in your working directory
3. They're already high resolution (300 DPI)
4. Just insert them directly into your report

### Best Practices:
- ✅ Use high resolution (300 DPI minimum)
- ✅ Make text readable (don't shrink too small)
- ✅ Crop out unnecessary parts
- ✅ Add captions below each screenshot
- ✅ Reference screenshots in text (e.g., "As shown in Figure 1...")

---

## 🚨 COMMON MISTAKES TO AVOID

1. ❌ **Missing the preprocessed data table** - This is explicitly required!
2. ❌ **No KL divergence interpretation** - Must include written analysis
3. ❌ **Incomplete manual ratings** - Need all 9 tables with ratings 1-5
4. ❌ **Wrong DCGAN grid size** - Must be 8×4 (32 images minimum)
5. ❌ **Low resolution images** - Use 300 DPI for all screenshots
6. ❌ **Not showing first few epochs** - Assignment requires early training progress
7. ❌ **Missing learning curves** - Confusion matrices and learning curves required

---

## ✅ FINAL VERIFICATION

Before submission, open your PDF report and verify:

1. **Can you see all 11 screenshots clearly?**
2. **Are all 4 PNG image files embedded?**
3. **Did you complete all 9 manual rating tables?**
4. **Is each screenshot referenced in the text?**
5. **Do all images have captions?**
6. **Is the STUDENT_ID updated in the BONUS section?**

If you can answer YES to all these questions, your documentation is complete! 🎉

---

**End of Screenshot Guide**
