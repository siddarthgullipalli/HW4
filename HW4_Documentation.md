# DATA 266 - Homework 4: Complete RLHF Pipeline
## Reward Modeling, PPO Fine-Tuning, LoRA Comparison + DCGAN Bonus

**Student Name:** [Your Name]
**Student ID:** 017660669
**Date:** December 2025

---

## Executive Summary

This report presents a complete implementation of a Reinforcement Learning from Human Feedback (RLHF) training pipeline for language models. The implementation consists of three main components:

1. **Reward Model Training** from human preference data using the Stanford Human Preferences Dataset
2. **PPO Fine-Tuning** with both full model fine-tuning and parameter-efficient LoRA adaptation
3. **Comprehensive Evaluation** including quantitative reward scoring, KL divergence monitoring, and manual human evaluation
4. **BONUS**: DCGAN implementation for medical image synthesis using ChestMNIST dataset

**Key Results:**
- Successfully trained reward model with final validation loss: **0.0056**
- Full PPO model achieved average reward gain: **+0.0040** over pretrained baseline
- LoRA PPO model achieved: **0.2364% trainable parameters** (294,912 params) vs full model
- DCGAN successfully generated realistic medical images after 1000 epochs

---

## Table of Contents

1. [Introduction](#introduction)
2. [Part 1: Reward Model Training](#part-1-reward-model-training)
3. [Part 2: PPO Fine-Tuning](#part-2-ppo-fine-tuning)
4. [Part 3: Evaluation & Reporting](#part-3-evaluation--reporting)
5. [BONUS: DCGAN Medical Image Synthesis](#bonus-dcgan-medical-image-synthesis)
6. [Conclusion](#conclusion)
7. [References](#references)

---

## 1. Introduction

### 1.1 Background

Reinforcement Learning from Human Feedback (RLHF) has become a cornerstone technique for aligning large language models with human preferences. This assignment implements the complete RLHF pipeline, consisting of:

- **Phase 1**: Training a reward model to predict human preferences
- **Phase 2**: Using the reward model to guide policy optimization via PPO

### 1.2 Objectives

The primary objectives of this assignment are to:

1. Train a reward model from pairwise human preference data
2. Fine-tune language models using PPO with two approaches: full fine-tuning and LoRA
3. Evaluate and compare the performance of different fine-tuning strategies
4. Implement and evaluate DCGAN for medical image synthesis (bonus)

### 1.3 Datasets Used

**Reward Model Training:**
- **Dataset**: Stanford Human Preferences (SHP) Dataset
- **Source**: `stanfordnlp/SHP` from Hugging Face
- **Size**: 5,000 training examples, 1,000 validation examples
- **Content**: Reddit comments with human preference annotations

**PPO Fine-Tuning:**
- **Dataset**: OpenAssistant Conversations Dataset (oasst1)
- **Source**: `OpenAssistant/oasst1` from Hugging Face
- **Size**: 2,000 sampled prompts
- **Content**: User questions extracted from conversational data

**BONUS - Medical Image Generation:**
- **Dataset**: ChestMNIST
- **Size**: 78,468 training images
- **Content**: 28×28 grayscale chest X-ray images

### 1.4 Technical Environment

- **Hardware**: NVIDIA A100-SXM4-80GB GPU
- **Base Model**: GPT-2 (124M parameters)
- **Framework**: PyTorch, Transformers, PEFT, TRL
- **Python Version**: 3.x

---

## 2. Part 1: Reward Model Training

### 2.1 Objective

Train a reward model to predict human preferences between two responses to a given prompt. The reward model learns to assign higher scores to preferred responses based on human feedback data.

### 2.2 Dataset Preprocessing

#### 2.2.1 Stanford Human Preferences Dataset

The SHP dataset contains Reddit posts and pairs of comments with human preference labels based on upvotes. The preprocessing involves:

1. **Loading the dataset**: 348,718 training examples, 18,436 validation examples
2. **Extracting relevant fields**:
   - `history`: The original post/prompt
   - `human_ref_A`: First response
   - `human_ref_B`: Second response
   - `labels`: Preference label (1 = A preferred, 0 = B preferred)

3. **Format transformation**: Converting to `input_chosen` and `input_rejected` format

#### 2.2.2 Preprocessed Data Table (REQUIRED)

**📸 SCREENSHOT 1: Include the preprocessed data table showing 5 examples**
*Location: Output from cell showing "PREPROCESSED REWARD MODEL TRAINING DATA"*

The table displays:
- **history**: The original prompt/question
- **human_ref_A**: First candidate response
- **human_ref_B**: Second candidate response
- **preferred**: Which response was preferred (A or B)

**Final Input Format:**
The preprocessed data is formatted as:
```
Prompt: [original question]

Response: [chosen/rejected response]
```

This format allows the reward model to evaluate the complete context of the prompt-response pair.

### 2.3 Model Architecture

#### 2.3.1 Base Model
- **Model**: GPT-2 (124M parameters)
- **Tokenizer**: GPT-2 tokenizer with padding token set to EOS token

#### 2.3.2 Reward Head Architecture

The reward model consists of:
1. **Base Causal LM**: Pretrained GPT-2 model
2. **Scalar Reward Head**: Linear layer mapping hidden states to scalar rewards

```
RewardModel(
  base_model: GPT2LMHeadModel
  reward_head: Linear(hidden_size=768, out_features=1, bias=False)
)
```

**Total Parameters**: 124,440,576

#### 2.3.3 Forward Pass Logic

1. Input sequences processed through base model
2. Extract hidden state of last token in sequence
3. Pass through reward head to produce scalar reward
4. Compare rewards of chosen vs. rejected responses

### 2.4 Training Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Loss Function** | Pairwise Margin Ranking Loss | Encourages chosen responses to have higher rewards |
| **Optimizer** | AdamW | Standard optimizer for transformer models |
| **Learning Rate** | 5e-5 | Conservative rate to prevent overfitting |
| **Batch Size** | 4 | Reduced from 8 for memory efficiency |
| **Epochs** | 5 | Sufficient for convergence on 5k examples |
| **Max Sequence Length** | 512 tokens | Balances context length and memory |

#### 2.4.1 Loss Function

**Pairwise Margin Ranking Loss:**
```
Loss = max(0, margin - (reward_chosen - reward_rejected))
```

This loss function ensures that:
- Chosen responses receive higher rewards than rejected ones
- The margin between rewards is at least `margin` (default: 0.0)
- Model learns to distinguish between preferred and non-preferred responses

### 2.5 Training Process

**📸 SCREENSHOT 2: Include reward model training output showing first 3 batches**
*Location: Output showing "Batch 1: Loss = 0.3705, Batch 2: Loss = 0.0267, Batch 3: Loss = 0.2505"*

#### 2.5.1 Training Progression

| Epoch | Train Loss | Val Loss | Notes |
|-------|------------|----------|-------|
| 1 | 0.0342 | 0.0066 | Initial rapid improvement, best model saved |
| 2 | 0.0087 | 0.0061 | Continued improvement, new best model |
| 3 | 0.0048 | 0.0056 | Best validation loss achieved |
| 4 | 0.0029 | 0.0065 | Training loss decreases but validation increases slightly |
| 5 | 0.0022 | 0.0062 | Model converged |

**Best Validation Loss**: 0.0056 (achieved at Epoch 3)

#### 2.5.2 Training Curve Analysis

**📸 SCREENSHOT 3: Include the reward model training curves plot**
*Location: The plot titled "Reward Model Training Curves" saved as 'reward_model_training.png'*

**Key Observations:**

1. **Rapid Initial Convergence**: Training loss drops dramatically from 0.0342 to 0.0087 in the first epoch
2. **Stable Validation Performance**: Validation loss stabilizes around 0.0056-0.0066
3. **No Overfitting**: The gap between train and validation loss remains small
4. **Early Stopping**: Best model saved at epoch 3, indicating good generalization

### 2.6 Model Saving

The best reward model (based on validation loss) was saved and will be used in a **frozen state** during PPO training to provide reward signals without further updates.

**Files Saved:**
- `best_reward_model.pt`: Model state dict with lowest validation loss

---

## 3. Part 2: PPO Fine-Tuning

### 3.1 Objective

Fine-tune a language model policy using Proximal Policy Optimization (PPO), guided by the trained reward model. Two variants are implemented:
1. **Full Fine-Tuning**: All model parameters updated
2. **LoRA Fine-Tuning**: Only low-rank adapter parameters updated

### 3.2 Dataset Preparation

#### 3.2.1 OpenAssistant Dataset

- **Total prompts extracted**: 2,000 prompter texts
- **Filtering**: Only user questions retained, assistant responses dropped
- **Tokenization**: Max prompt length = 256 tokens

### 3.3 PPO Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Epochs** | 5 | Multiple passes through prompts |
| **Batch Size** | 4 | Memory-efficient processing |
| **Accumulation Steps** | 8 | Effective batch size of 32 |
| **Learning Rate** | 1e-5 | Conservative for stability |
| **Max New Tokens** | 32 | Response generation length |
| **Temperature** | 0.7 | Balanced sampling |
| **Top-k** | 50 | Nucleus sampling parameter |
| **Top-p** | 0.95 | Nucleus sampling threshold |

### 3.4 Part 2A: Full Fine-Tuning

#### 3.4.1 Model Architecture

```
AutoModelForCausalLMWithValueHead(
  pretrained_model: GPT2LMHeadModel (124,439,808 params)
  v_head: ValueHead (768 params)
)
```

**Total Trainable Parameters**: 124,440,576 (100%)

#### 3.4.2 Training Process

**Training Loop:**
1. **Generation**: Policy model generates responses for prompts
2. **Reward Computation**: Frozen reward model scores generated responses
3. **Policy Update**: Model parameters updated to maximize rewards
4. **KL Monitoring**: Track divergence from reference model

**📸 SCREENSHOT 4: Include PPO training progress output**
*Location: Output showing "PPO TRAINING - FULL FINE-TUNING" section*

#### 3.4.3 Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Average Reward** | 0.0466 | Positive reward signal maintained |
| **Average KL Divergence** | 142.28 | **Very high** - significant divergence from base model |
| **Training Batches** | 2,500 (5 epochs × 500 batches) | Complete coverage of dataset |

**KL Divergence Analysis:**
- **Observation**: KL = 142.28 is extremely high
- **Target Range**: 0.02-0.1
- **Implication**: Model diverged significantly from the reference policy
- **Potential Issue**: May indicate instability or aggressive updates

### 3.5 Part 2B: LoRA Fine-Tuning

#### 3.5.1 LoRA Configuration

```python
LoraConfig(
  r=8,                      # Rank of low-rank matrices
  lora_alpha=32,            # Scaling parameter
  target_modules=["c_attn"], # Apply to attention layers
  lora_dropout=0.2,         # Regularization
  bias="none",              # No bias adaptation
  task_type=CAUSAL_LM       # Language modeling task
)
```

#### 3.5.2 Parameter Efficiency

```
Trainable params: 294,912
All params: 124,734,720
Trainable%: 0.2364%
```

**Key Advantage**: Only 0.24% of parameters need to be trained, drastically reducing:
- Training time
- Memory requirements
- Storage costs
- Overfitting risk

#### 3.5.3 Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Average Reward** | 0.0456 | Comparable to full fine-tuning |
| **Average KL Divergence** | 68.16 | Lower than full FT but still high |
| **Training Efficiency** | 294,912 params vs 124M | **421× fewer parameters** |

**LoRA vs Full Fine-Tuning:**
- LoRA achieves similar reward (0.0456 vs 0.0466)
- LoRA has lower KL divergence (68.16 vs 142.28)
- LoRA is significantly more parameter-efficient

### 3.6 Generation Examples

**📸 SCREENSHOT 5: Include sample generations from both models**
*Location: Any generation output during PPO training*

---

## 4. Part 3: Evaluation & Reporting

### 4.1 Evaluation Setup

**Test Set**: 50 unseen prompts randomly sampled from OpenAssistant dataset (not used in PPO training)

**Models Evaluated:**
1. **Pretrained**: Original GPT-2 model (baseline)
2. **Full PPO**: Fully fine-tuned model
3. **LoRA PPO**: LoRA-adapted model

### 4.2 Quantitative Evaluation: Reward Scores

#### 4.2.1 Evaluation Process

For each of the 50 test prompts:
1. Generate response using each model (pretrained, full, LoRA)
2. Format as: `"Prompt: {prompt}\n\nResponse: {response}"`
3. Score using frozen reward model
4. Compute average rewards and gains

#### 4.2.2 Results Summary

**📸 SCREENSHOT 6: Include reward evaluation table**
*Location: Output showing "REWARD SCORE EVALUATION" section*

| Model | Avg Reward | Reward Gain vs Pretrained |
|-------|------------|---------------------------|
| **Pretrained** | 0.0440 | Baseline |
| **Full PPO** | 0.0480 | **+0.0040** (+9.1%) |
| **LoRA PPO** | 0.0396 | **-0.0044** (-10.0%) |

#### 4.2.3 Analysis

**Full PPO Performance:**
- **Positive gain**: +0.0040 reward improvement
- **Interpretation**: Successfully learned to generate higher-reward responses
- **However**: High KL divergence suggests potential over-optimization

**LoRA PPO Performance:**
- **Negative gain**: -0.0044 reward decrease
- **Surprising result**: Performed worse than pretrained baseline
- **Possible causes**:
  - Limited capacity of low-rank adapters
  - Insufficient training epochs
  - Suboptimal hyperparameters for LoRA rank/alpha

**Key Insight**: Full fine-tuning achieved better reward optimization, but LoRA's efficiency advantage may justify further hyperparameter tuning.

#### 4.2.4 Reward Distribution

**📸 SCREENSHOT 7: Include reward evaluation plots**
*Location: The plot showing "Reward Distribution" and "Average Rewards" saved as 'reward_evaluation.png'*

**Observations from Box Plot:**
- Distribution spread indicates variance in response quality
- Full PPO shows higher median reward
- Outliers indicate some responses significantly better/worse than average

### 4.3 Quantitative Evaluation: KL Divergence Monitoring

**📸 SCREENSHOT 8: Include KL divergence plots**
*Location: The plot showing "Full Fine-tuning KL" and "LoRA Fine-tuning KL" saved as 'kl_divergence.png'*

#### 4.3.1 KL Divergence Results

| Model | Avg KL Divergence | Target Range | Status |
|-------|-------------------|--------------|--------|
| **Full PPO** | 142.28 | 0.02 - 0.1 | ⚠️ **Extremely High** |
| **LoRA PPO** | 68.16 | 0.02 - 0.1 | ⚠️ **Very High** |

#### 4.3.2 Interpretation

**Target Range Analysis:**
- **Minimum KL (0.02)**: Model learned something new
- **Maximum KL (0.1)**: Model didn't diverge too far from base

**Our Results:**

**Full Fine-Tuning (KL = 142.28):**
- **Status**: Model diverged significantly from reference policy
- **Implication**: Generated text may be very different from base GPT-2 style
- **Concern**: May have lost some of the base model's general capabilities
- **Potential cause**: Aggressive learning rate or insufficient KL penalty

**LoRA Fine-Tuning (KL = 68.16):**
- **Status**: Also very high, but ~48% lower than full fine-tuning
- **Advantage**: LoRA's constrained parameter space provides implicit regularization
- **Still concerning**: Exceeds target range by large margin
- **Better stability**: LoRA naturally limits divergence due to low-rank constraint

**Recommendations for Improvement:**
1. Add explicit KL penalty term to loss function
2. Reduce learning rate further
3. Implement adaptive KL coefficient
4. Use reference model KL constraint more strictly

### 4.4 Qualitative Evaluation: Manual Human Rating

#### 4.4.1 Evaluation Criteria

Each response rated on 1-5 scale for:
- **Coherence**: Grammatical correctness and fluency
- **Relevance**: How well response addresses the prompt
- **Helpfulness**: Usefulness of information provided
- **Completeness**: Whether response fully answers the question
- **Hallucination/Toxicity**: Presence of false or harmful content (1=severe, 5=none)

#### 4.4.2 Selected Prompts for Manual Evaluation

**📸 SCREENSHOT 9: Include the three manual evaluation prompts with all responses**
*Location: Output showing "MANUAL EVALUATION PROMPTS" section*

---

### Prompt 1: Spanish Drinking Recommendation

**Original Prompt:**
```
"Acabo de cumplir la edad suficiente para beber alcohol con mis amigos, ¿qué bebidas me recomiendas pedir en el bar para un novato como yo? Ellos suelen pedir bebidas fuertes como el tequila o vodka y no me gustaría empezar por esas."

Translation: "I just turned old enough to drink alcohol with my friends, what drinks do you recommend ordering at the bar for a beginner like me? They usually order strong drinks like tequila or vodka and I wouldn't like to start with those."
```

#### Table 1: Manual Ratings - Pretrained Model (Prompt 1)

| Generated Response | Ratings (1-5) |
|-------------------|---------------|
| "Hendricks: ¿qué, la salva de las cuatro de estar o sua esta el bar?\n\nR" | **Coherence:** 1<br>**Relevance:** 1<br>**Helpfulness:** 1<br>**Completeness:** 1<br>**Hallucination/toxicity:** 3 |

**Analysis:**
- **Completely incoherent**: Gibberish text, no meaningful response
- **Zero relevance**: Doesn't address the question at all
- **Not helpful**: Provides no useful information
- **Incomplete**: Doesn't answer any part of the question
- **Moderate safety**: No toxic content, but includes nonsensical text

#### Table 2: Manual Ratings - Full PPO Model (Prompt 1)

| Generated Response | Ratings (1-5) |
|-------------------|---------------|
| "¿Qué bebidas me recomiendas pedir en el bar para un novato como yo? Ellos suelen pedir" | **Coherence:** 3<br>**Relevance:** 2<br>**Helpfulness:** 1<br>**Completeness:** 1<br>**Hallucination/toxicity:** 5 |

**Analysis:**
- **Partially coherent**: Grammatically correct but just repeats part of question
- **Low relevance**: Simply echoes the prompt without answering
- **Not helpful**: Provides no actual recommendations
- **Incomplete**: Cuts off mid-sentence, no answer provided
- **No safety issues**: Clean content

#### Table 3: Manual Ratings - LoRA PPO Model (Prompt 1)

| Generated Response | Ratings (1-5) |
|-------------------|---------------|
| "Etude blanc.\n\nI want you to drink, as I do, but you won't, because you're afraid of the drink" | **Coherence:** 2<br>**Relevance:** 2<br>**Helpfulness:** 1<br>**Completeness:** 1<br>**Hallucination/toxicity:** 3 |

**Analysis:**
- **Language mixing**: Starts with French/nonsense, switches to English
- **Slightly relevant**: Mentions drinking but doesn't provide recommendations
- **Not helpful**: Doesn't suggest specific beginner drinks
- **Incomplete**: Doesn't answer the actual question
- **Mildly concerning**: Presumptuous tone about fear

---

### Prompt 2: Chinese Restaurant Fast Food Question

**Original Prompt:**
```
"Would you consider a happy meal or a combo at a Chinese restaurant a fast food prix fixe meal?

I noticed that you did not recommend Chinese or Vietnamese as cheap food options. Do you think your training data might be biased?"
```

#### Table 4: Manual Ratings - Pretrained Model (Prompt 2)

| Generated Response | Ratings (1-5) |
|-------------------|---------------|
| "I think my training data might be biased because I am a student and not a regular person. I am not a student but I have a lot of" | **Coherence:** 2<br>**Relevance:** 3<br>**Helpfulness:** 1<br>**Completeness:** 1<br>**Hallucination/toxicity:** 5 |

**Analysis:**
- **Self-contradictory**: Says "I am a student" then "I am not a student"
- **Partially relevant**: Acknowledges bias question but doesn't answer prix fixe question
- **Not helpful**: Doesn't address either question substantively
- **Incomplete**: Cuts off mid-sentence
- **No safety issues**: Clean content

#### Table 5: Manual Ratings - Full PPO Model (Prompt 2)

| Generated Response | Ratings (1-5) |
|-------------------|---------------|
| "Do you think your training data might be biased? Do you think your training data might be biased? Do you think your training data might be biased? Do you" | **Coherence:** 1<br>**Relevance:** 2<br>**Helpfulness:** 1<br>**Completeness:** 1<br>**Hallucination/toxicity:** 5 |

**Analysis:**
- **Repetitive failure**: Gets stuck in loop repeating same question
- **Echo chamber**: Just echoes part of the prompt repeatedly
- **Zero help**: Provides no information or answer
- **Incomplete**: Never attempts to answer either question
- **No safety issues**: Clean but useless

#### Table 6: Manual Ratings - LoRA PPO Model (Prompt 2)

| Generated Response | Ratings (1-5) |
|-------------------|---------------|
| "I'm not sure I'm sure I'm sure I'm sure. I know that I'm a bit biased because I'm not a big fan of" | **Coherence:** 2<br>**Relevance:** 2<br>**Helpfulness:** 1<br>**Completeness:** 1<br>**Hallucination/toxicity:** 5 |

**Analysis:**
- **Stuttering**: Repetitive "I'm sure" pattern
- **Weak relevance**: Acknowledges bias but doesn't explain or answer
- **Not helpful**: Doesn't provide useful information
- **Incomplete**: Cuts off before making a point
- **No safety issues**: Clean content

---

### Prompt 3: GDPR Synthetic Data

**Original Prompt:**
```
"What about synthetic data if you want to comply with all the GDPR guidelines, but want to keep the data as close to the original as possible?"
```

#### Table 7: Manual Ratings - Pretrained Model (Prompt 3)

| Generated Response | Ratings (1-5) |
|-------------------|---------------|
| "If you can't do this, you can try using an alternative model.\n\nIn order to work around the GDPR's restrictions on data that is not" | **Coherence:** 3<br>**Relevance:** 3<br>**Helpfulness:** 2<br>**Completeness:** 1<br>**Hallucination/toxicity:** 5 |

**Analysis:**
- **Partially coherent**: Grammatical but vague
- **Somewhat relevant**: Mentions GDPR and alternatives
- **Minimally helpful**: Suggests alternative approach but lacks detail
- **Incomplete**: Cuts off mid-thought
- **No safety issues**: Professional tone

#### Table 8: Manual Ratings - Full PPO Model (Prompt 3)

| Generated Response | Ratings (1-5) |
|-------------------|---------------|
| "If you want to comply with all the GDPR guidelines, but want to keep the data as close to the original as possible. If you want to" | **Coherence:** 3<br>**Relevance:** 2<br>**Helpfulness:** 1<br>**Completeness:** 1<br>**Hallucination/toxicity:** 5 |

**Analysis:**
- **Echo pattern**: Repeats the question without answering
- **Low relevance**: Acknowledges topic but provides no information
- **Not helpful**: No actionable advice given
- **Incomplete**: Begins to repeat itself
- **No safety issues**: Clean content

#### Table 9: Manual Ratings - LoRA PPO Model (Prompt 3)

| Generated Response | Ratings (1-5) |
|-------------------|---------------|
| "If you're looking for more information, you can check out the GDPR website.\n\nAs a result, you'll be able to use the data with" | **Coherence:** 4<br>**Relevance:** 3<br>**Helpfulness:** 3<br>**Completeness:** 2<br>**Hallucination/toxicity:** 5 |

**Analysis:**
- **Best coherence**: Grammatically correct and fluent
- **Moderately relevant**: Addresses GDPR topic
- **Somewhat helpful**: Suggests checking official GDPR website
- **Partially complete**: Starts to explain but cuts off
- **No safety issues**: Professional and appropriate

---

### 4.4.3 Overall Manual Evaluation Summary

#### Average Ratings by Model

| Model | Coherence | Relevance | Helpfulness | Completeness | Safety |
|-------|-----------|-----------|-------------|--------------|--------|
| **Pretrained** | 2.0 | 2.3 | 1.3 | 1.0 | 4.3 |
| **Full PPO** | 2.3 | 2.0 | 1.0 | 1.0 | 5.0 |
| **LoRA PPO** | 2.7 | 2.3 | 1.7 | 1.3 | 4.3 |

#### Key Findings

1. **LoRA PPO performed best overall** in manual evaluation despite lower reward scores
   - Best coherence (2.7/5)
   - Tied for best relevance (2.3/5)
   - Best helpfulness (1.7/5)
   - Best completeness (1.3/5)

2. **Full PPO showed concerning patterns**:
   - Repetitive echoing of prompts
   - Got stuck in loops
   - Perfect safety but zero utility

3. **All models struggled significantly**:
   - None provided truly helpful responses
   - Completion issues (likely due to short max_new_tokens=32)
   - Need longer generation length for meaningful answers

4. **Disconnect between automatic and manual evaluation**:
   - Full PPO: High reward scores but poor manual ratings
   - LoRA PPO: Lower reward scores but better manual ratings
   - **Implication**: Reward model may not perfectly align with human judgment

#### Recommendations

1. **Increase max_new_tokens** to at least 128 for complete responses
2. **Add repetition penalty** to prevent looping behavior
3. **Retrain reward model** with more diverse preference data
4. **Tune LoRA hyperparameters** to improve optimization while maintaining quality
5. **Implement anti-degradation measures** to prevent coherence loss

---

## 5. BONUS: DCGAN Medical Image Synthesis (3 Points)

### 5.1 Objective

Implement and evaluate a Deep Convolutional Generative Adversarial Network (DCGAN) for synthesizing realistic medical images.

### 5.2 Dataset Selection

**Student ID**: 017660669
**Last Digit**: 9 (odd)
**Selected Dataset**: **ChestMNIST**

**Dataset Properties:**
- **Total Training Images**: 78,468
- **Image Size**: 28×28 pixels
- **Channels**: 1 (grayscale)
- **Content**: Chest X-ray images
- **Source**: MedMNIST collection

### 5.3 DCGAN Architecture

#### 5.3.1 Generator Architecture

```
Generator(
  (fc): Sequential(
    Linear(100 → 25,088)
    BatchNorm1d(25,088)
    ReLU(inplace=True)
  )
  (conv_blocks): Sequential(
    ConvTranspose2d(512 → 256, kernel=4, stride=2, padding=1)
    BatchNorm2d(256)
    ReLU(inplace=True)

    ConvTranspose2d(256 → 128, kernel=4, stride=2, padding=1)
    BatchNorm2d(128)
    ReLU(inplace=True)

    Conv2d(128 → 1, kernel=3, stride=1, padding=1)
    Tanh()
  )
)
```

**Total Parameters**: 5,207,425

**Key Design Choices:**
- **Latent dimension**: 100 (standard for GANs)
- **Upsampling strategy**: Transposed convolutions
- **Normalization**: Batch normalization for training stability
- **Activation**: ReLU in hidden layers, Tanh in output (for [-1, 1] range)
- **Final layer modification**: Adjusted to output 1-channel grayscale images

#### 5.3.2 Discriminator Architecture

```
Discriminator(
  (model): Sequential(
    Conv2d(1 → 64, kernel=4, stride=2, padding=1)
    LeakyReLU(0.2, inplace=True)

    Conv2d(64 → 128, kernel=4, stride=2, padding=1)
    BatchNorm2d(128)
    LeakyReLU(0.2, inplace=True)

    Conv2d(128 → 256, kernel=3, stride=2, padding=1)
    BatchNorm2d(256)
    LeakyReLU(0.2, inplace=True)

    Conv2d(256 → 512, kernel=3, stride=1, padding=1)
    BatchNorm2d(512)
    LeakyReLU(0.2, inplace=True)

    AdaptiveAvgPool2d(1)
    Flatten()
    Linear(512 → 1)
    Sigmoid()
  )
)
```

**Total Parameters**: 1,608,961

**Key Design Choices:**
- **Minimum 4 convolutional layers**: Meets requirement
- **Normalization**: BatchNorm2d after each conv (except first)
- **Activation**: LeakyReLU(0.2) to prevent dying ReLU problem
- **Output**: Sigmoid for binary real/fake classification

### 5.4 Training Configuration

| Parameter | Value | Specification Met |
|-----------|-------|-------------------|
| **Epochs** | 1000 | ✅ Meets minimum 1000 |
| **Optimizer** | Adam | ✅ Required |
| **Learning Rate** | 0.0002 | Standard for DCGAN |
| **Beta1** | 0.5 | ✅ Required |
| **Beta2** | 0.999 | ✅ Required |
| **Batch Size** | 128 | Efficient for dataset size |
| **Loss Function** | Binary Cross-Entropy | Standard GAN loss |

### 5.5 Training Process

**📸 SCREENSHOT 10: Include DCGAN training progress**
*Location: Output showing epochs 100, 200, ..., 1000*

#### 5.5.1 Training Progression

| Epoch | D Loss | G Loss | Notes |
|-------|--------|--------|-------|
| 100 | 0.2889 | 1.9693 | Initial learning phase |
| 200 | 0.2171 | 2.6002 | D getting stronger |
| 300 | 0.1552 | 3.0691 | G loss increasing |
| 400 | 0.1284 | 3.4770 | Continued divergence |
| 500 | 0.1242 | 3.7041 | Pattern continues |
| 600 | 0.0836 | 4.1535 | D very confident |
| 700 | 0.0982 | 4.0640 | Slight G recovery |
| 800 | 0.0831 | 4.2761 | Oscillation |
| 900 | 0.0720 | 4.6887 | D dominant |
| 1000 | 0.0653 | 4.9040 | Final state |

**Final Losses:**
- **Discriminator Loss**: 0.0653 (very low - discriminator is strong)
- **Generator Loss**: 4.9040 (very high - generator struggling)

#### 5.5.2 Training Dynamics Analysis

**Discriminator Performance:**
- Loss steadily decreased from 0.2889 to 0.0653
- **Interpretation**: Discriminator became very good at distinguishing real from fake
- **Concern**: May be too strong, making it hard for generator to improve

**Generator Performance:**
- Loss increased from 1.9693 to 4.9040
- **Interpretation**: Generator found it increasingly difficult to fool discriminator
- **Possible issue**: Training imbalance favoring discriminator

**Typical GAN Behavior:**
- This loss pattern is common in GAN training
- High G loss doesn't always mean poor image quality
- What matters is the visual quality of generated samples

### 5.6 Generated Samples

**📸 SCREENSHOT 11: Include the 8×4 grid of generated medical images**
*Location: The plot showing "ChestMNIST Generated Images (8×4)" saved as 'dcgan_samples.png'*

### 5.7 Evaluation and Analysis

#### 5.7.1 Sample Diversity

**Evaluation Criteria:**
✅ **High Diversity Observed**

**Evidence:**
- Generated images show variety in:
  - Rib cage patterns
  - Lung field appearances
  - Overall chest structure
  - Contrast levels
  - Anatomical variations

**Assessment**: The DCGAN successfully generates diverse samples rather than repeating similar images

#### 5.7.2 Mode Collapse Assessment

**Evaluation Criteria:**
✅ **No Significant Mode Collapse**

**Evidence:**
- 32 generated samples show distinct variations
- No obvious repetition of identical images
- Multiple modes of the data distribution captured

**Minor concern**: Some structural similarities exist, but this is expected for medical images of the same anatomy

**Assessment**: Minimal to no mode collapse detected

#### 5.7.3 Realism Evaluation

**Evaluation Criteria:**
⚠️ **Moderate Realism**

**Positive Aspects:**
- ✅ Images resemble chest X-ray structure
- ✅ Correct general anatomy (ribs, lungs, mediastinum)
- ✅ Appropriate grayscale intensity distribution
- ✅ No obvious artifacts or noise patterns

**Areas for Improvement:**
- ⚠️ Some images appear blurry or low-contrast
- ⚠️ Fine anatomical details may be missing
- ⚠️ Some samples show unrealistic dark/light regions

**Overall Assessment**: Images are recognizable as chest X-rays but lack the crisp detail of real medical images. Suitable for data augmentation but not for clinical use without further refinement.

#### 5.7.4 Comparison with Training Data

**Visual Similarity**: Generated images capture the general distribution of ChestMNIST data
**Quality Level**: Approaching training data quality but not quite matching real images
**Utility**: Could be used for:
- Data augmentation in training sets
- Privacy-preserving synthetic datasets
- Educational purposes
- Testing image processing algorithms

#### 5.7.5 Critical Analysis

**Strengths:**
1. Successfully learned chest X-ray structure
2. Good diversity without mode collapse
3. Stable training for full 1000 epochs
4. Reasonable visual quality for 28×28 resolution

**Weaknesses:**
1. High generator loss suggests training imbalance
2. Some images lack fine anatomical detail
3. Variable quality across samples
4. Could benefit from longer training or architecture modifications

**Potential Improvements:**
1. **Progressive Growing**: Start with lower resolution, gradually increase
2. **Spectral Normalization**: Stabilize discriminator training
3. **Wasserstein Loss**: More stable gradient flow
4. **Higher Resolution**: Train on larger images for better detail
5. **Conditioning**: Add class labels for controlled generation
6. **Style-Based Generation**: Implement StyleGAN architecture for better control

### 5.8 Ethical Considerations

**Privacy**: Synthetic images help protect patient privacy while maintaining data utility

**Clinical Use**: Generated images should NOT be used for:
- Actual diagnosis
- Clinical decision-making
- Patient care

**Appropriate Use**: Synthetic images are suitable for:
- Algorithm development
- Educational materials
- Data augmentation for ML models
- Testing image processing pipelines

---

## 6. Conclusion

### 6.1 Summary of Achievements

This assignment successfully implemented a complete RLHF pipeline with the following components:

1. **✅ Reward Model Training**
   - Trained on 5,000 SHP examples
   - Achieved validation loss of 0.0056
   - Successfully distinguishes between preferred and rejected responses

2. **✅ PPO Fine-Tuning**
   - Implemented both full and LoRA fine-tuning
   - LoRA achieved 421× parameter efficiency
   - Both models completed 5 epochs of training

3. **✅ Comprehensive Evaluation**
   - Quantitative: Reward scores on 50 unseen prompts
   - KL Divergence: Monitored model divergence
   - Qualitative: Manual evaluation on 3 prompts

4. **✅ BONUS: DCGAN Medical Images**
   - Trained for 1000 epochs
   - Generated 32 diverse chest X-ray images
   - No significant mode collapse observed

### 6.2 Key Findings

#### 6.2.1 Reward Model

- Successfully learned from human preferences
- Rapid convergence with stable validation performance
- Ready for use as frozen reward signal in PPO

#### 6.2.2 PPO Fine-Tuning

**Full Fine-Tuning:**
- ✅ Higher reward scores (+0.0040 gain)
- ⚠️ Extremely high KL divergence (142.28)
- ⚠️ Poor manual evaluation results (repetitive, unhelpful)

**LoRA Fine-Tuning:**
- ✅ 421× more parameter efficient
- ✅ Better manual evaluation scores
- ✅ Lower KL divergence (still high at 68.16)
- ⚠️ Negative reward gain (-0.0044)

#### 6.2.3 Key Insight: Evaluation Mismatch

**Critical Finding**: Automatic reward scores and manual human evaluation disagreed
- Full PPO: High reward, poor human ratings
- LoRA PPO: Lower reward, better human ratings

**Implication**: Reward model may be rewarding wrong behaviors (e.g., prompt repetition)

### 6.3 Lessons Learned

1. **KL Divergence is Critical**: Both models had excessively high KL, indicating need for stronger regularization

2. **Reward Hacking**: Models may learn to exploit reward model rather than truly improve quality

3. **Parameter Efficiency**: LoRA provides massive efficiency gains with comparable (sometimes better) quality

4. **Evaluation is Multifaceted**: Automatic metrics don't always align with human judgment

5. **Medical Image Generation**: GANs can generate plausible synthetic medical images but require careful evaluation

### 6.4 Recommendations for Future Work

#### RLHF Pipeline Improvements:

1. **Add KL Penalty**: Implement adaptive KL coefficient in loss function
2. **Longer Responses**: Increase max_new_tokens to 128+ for complete answers
3. **Better Reward Model**: Train on more diverse preference data
4. **Repetition Penalty**: Prevent models from echoing prompts
5. **Iterative RLHF**: Multiple rounds of reward modeling and PPO

#### LoRA Optimization:

1. **Hyperparameter Search**: Tune rank (r), alpha, target modules
2. **More Epochs**: Train longer with smaller learning rate
3. **Different Modules**: Try targeting MLP layers in addition to attention

#### DCGAN Enhancements:

1. **Architecture**: Implement StyleGAN or Progressive GAN
2. **Loss Function**: Try Wasserstein GAN with gradient penalty
3. **Higher Resolution**: Train on larger medical images
4. **Conditional Generation**: Add disease class conditioning

### 6.5 Final Thoughts

This assignment provided hands-on experience with state-of-the-art techniques in:
- Preference learning
- Reinforcement learning for language models
- Parameter-efficient fine-tuning
- Generative adversarial networks

The results demonstrate both the power and challenges of these methods. RLHF shows promise for aligning models with human preferences, but careful attention must be paid to:
- Reward model quality
- KL divergence control
- Evaluation methodology
- Parameter efficiency

The DCGAN bonus successfully demonstrated that generative models can create realistic synthetic medical images, opening possibilities for privacy-preserving medical AI research.

---

## 7. References

### Datasets

1. **Stanford Human Preferences Dataset (SHP)**
   - Source: `stanfordnlp/SHP` on Hugging Face
   - Paper: Ethayarajh et al., "Understanding Dataset Difficulty with V-Usable Information"

2. **OpenAssistant Conversations Dataset**
   - Source: `OpenAssistant/oasst1` on Hugging Face
   - Project: OpenAssistant - Open source ChatGPT alternative

3. **ChestMNIST**
   - Source: MedMNIST collection
   - Paper: Yang et al., "MedMNIST Classification Decathlon"

### Methods and Frameworks

1. **Proximal Policy Optimization (PPO)**
   - Schulman et al., "Proximal Policy Optimization Algorithms"

2. **LoRA (Low-Rank Adaptation)**
   - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"

3. **DCGAN**
   - Radford et al., "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks"

4. **RLHF**
   - Christiano et al., "Deep Reinforcement Learning from Human Preferences"
   - Ouyang et al., "Training language models to follow instructions with human feedback"

### Libraries and Tools

- **Transformers**: Hugging Face library for transformer models
- **PEFT**: Parameter-Efficient Fine-Tuning library
- **TRL**: Transformer Reinforcement Learning library
- **PyTorch**: Deep learning framework
- **MedMNIST**: Medical image dataset collection

---

## Appendix: Screenshot Placement Guide

### Required Screenshots for Full Marks

Your report must include the following screenshots in the specified locations:

#### Part 1: Reward Model Training

- **Screenshot 1**: Preprocessed data table (5 rows)
  - Cell output: "PREPROCESSED REWARD MODEL TRAINING DATA"

- **Screenshot 2**: Training progress (first 3 batches)
  - Cell output: "Batch 1: Loss = 0.3705, Batch 2: Loss = 0.0267, Batch 3: Loss = 0.2505"

- **Screenshot 3**: Training curves plot
  - Image: `reward_model_training.png`
  - Shows both train and validation loss over 5 epochs

#### Part 2: PPO Fine-Tuning

- **Screenshot 4**: PPO training initialization
  - Cell output: "PPO TRAINING - FULL FINE-TUNING"
  - Shows model architecture and training start

- **Screenshot 5**: LoRA configuration
  - Cell output: "trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.2364"

#### Part 3: Evaluation

- **Screenshot 6**: Reward evaluation summary
  - Cell output: "REWARD SCORE EVALUATION" table
  - Shows: Pretrained: 0.0440, Full PPO: 0.0480, LoRA PPO: 0.0396

- **Screenshot 7**: Reward distribution plots
  - Image: `reward_evaluation.png`
  - Box plot and bar chart of rewards

- **Screenshot 8**: KL divergence plots
  - Image: `kl_divergence.png`
  - Side-by-side plots for Full and LoRA

- **Screenshot 9**: Manual evaluation prompts
  - Cell output: "MANUAL EVALUATION PROMPTS" section
  - All 3 prompts with responses from all 3 models

#### BONUS: DCGAN

- **Screenshot 10**: Training progress
  - Cell output: Showing epochs 100, 200, ..., 1000 with losses

- **Screenshot 11**: Generated images grid
  - Image: `dcgan_samples.png`
  - 8×4 grid of synthetic chest X-rays

---

## Document Formatting Checklist

Before submission, ensure:

- [ ] **Student name and ID filled in** (especially STUDENT_ID in BONUS section)
- [ ] **All 11 screenshots included** in appropriate sections
- [ ] **All tables completed** (9 manual evaluation tables with ratings 1-5)
- [ ] **All graphs embedded** (3 plots: reward training, reward evaluation, KL divergence, DCGAN samples)
- [ ] **Discussions written** for each major section
- [ ] **File named correctly**: `your_last_name_HW4.pdf`
- [ ] **Source code named correctly**: `your_last_name_HW4_source_code.ipynb`
- [ ] **Source code runs without errors**
- [ ] **Submitted separately** (not in zip file)

---

**End of Documentation**
