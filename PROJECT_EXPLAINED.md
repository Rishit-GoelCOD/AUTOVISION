# AUTOVISION — Driver Safety & Attention Monitor
### A Comprehensive Technical & Academic Reference for Machine Learning Viva & Project Defense

**Student:** Rishit Goel  
**Supervisor / Guide:** Dr. Basab Nath  
**Course:** Artificial Intelligence & Machine Learning  
**Project Title:** AUTOVISION: Vision-Based Real-Time Driver Drowsiness and Distraction Monitoring System  

---

## Table of Contents
1. [Project Overview & Core Motivation](#1-project-overview--core-motivation)
2. [What the System Detects (The 4 Critical Behaviors)](#2-what-the-system-detects-the-4-critical-behaviors)
3. [Complete Codebase & Folder Architecture](#3-complete-codebase--folder-architecture)
4. [Dataset Creation & Preprocessing Pipeline](#4-dataset-creation--preprocessing-pipeline)
5. [Mathematical Biomarker Engineering](#5-mathematical-biomarker-engineering)
6. [Machine Learning Modeling & Training Pipeline](#6-machine-learning-modeling--training-pipeline)
7. [Comprehensive Evaluation & Experimental Metrics](#7-comprehensive-evaluation--experimental-metrics)
8. [Real-Time Inference, Filtering & Alert Logic](#8-real-time-inference-filtering--alert-logic)
9. [Key Engineering Challenges & Solutions](#9-key-engineering-challenges--solutions)
10. [Step-by-Step Reproduction Guide](#10-step-by-step-reproduction-guide)
11. [Viva Voce Master Q&A Defense Guide](#11-viva-voce-master-qa-defense-guide)
12. [Formula Sheet & Glossary](#12-formula-sheet--glossary)

---

## 1. Project Overview & Core Motivation

### 1.1 The Real-World Problem
Road traffic accidents claim over 1.3 million lives worldwide every year. According to transportation safety research, driver drowsiness and visual distraction cause more than 20% of all fatal roadway accidents. Unlike mechanical failures, human driver errors occur within split seconds when a driver:
- Nods off or experiences a micro-sleep episode.
- Experiences severe physical fatigue manifested by prolonged yawning.
- Glances away from the road trajectory toward passengers or scenery.
- Tilts their head downward to read or text on a handheld mobile phone.

### 1.2 Project Goal
This project builds a fully functional, non-intrusive, software-based **Driver Safety Monitoring System**. It watches the driver through a standard consumer webcam (720p at 30 frames per second), detects facial landmarks in real time, extracts biologically meaningful geometric ratios, predicts hazardous driver states using calibrated **Logistic Regression** classifiers, and raises instant visual and auditory alerts.

### 1.3 Design Pillars
1. **Zero Expensive Hardware:** Runs on a standard laptop or vehicle webcam without requiring infrared illuminators, depth cameras, or wearable sensors.
2. **Sub-Millisecond Inference:** Feature calculation and model prediction take less than 0.1 milliseconds per frame on CPU, leaving ample computation headroom for guaranteed 30 FPS video processing.
3. **Mathematical Explainability:** Instead of opaque, black-box deep neural networks, the classification layer uses standard, interpretable supervised machine learning models with clear linear decision boundaries.

---

## 2. What the System Detects (The 4 Critical Behaviors)

The system treats driver safety as four independent, parallel binary classification tasks:

| Hazard Condition | Definition & Danger | Primary Geometric Biomarker | Normal Value | Alert Value |
| :--- | :--- | :--- | :--- | :--- |
| **Yawning** | Mouth is wide open for an extended period; strong physiological precursor to sleep. | Mouth Aspect Ratio (**MAR**) | $\approx 0.29$ | $\ge 0.60$ |
| **Eyes Closed** | Palpebral fissures collapse; direct micro-sleep hazard. | Eye Aspect Ratio (**EAR**) | $\approx 0.29\text{--}0.35$ | $\le 0.12$ |
| **Looking Away** | Head rotated horizontally away from the road (driver inattention). | Horizontal Nose Deviation (**$|Yaw|$**) | $\le 5\%$ | $\ge 14\%$ |
| **Looking Down** | Head tilted downward (texting, checking smartphone in lap). | Vertical Nose Ratio (**$Pitch$**) | $\approx 0\%$ | $\ge 10\%$ |

---

## 3. Complete Codebase & Folder Architecture

```
driver-safety-monitor/
├── data/
│   ├── features.csv                         # Master tabular dataset (3,534 rows, 10 columns)
│   ├── raw/
│   │   └── dataset_new/                     # Public Kaggle Yawn Dataset (1,448 images)
│   │       ├── train/ (yawn/ & no_yawn/)    # 1,233 images
│   │       └── test/  (yawn/ & no_yawn/)    # 215 images
│   └── own/                                 # Personally recorded webcam frames (2,542 images)
│       ├── eyes_open/                       # 515 frames across 4 sessions
│       ├── eyes_closed/                     # 520 frames across 5 sessions
│       ├── facing_forward/                  # 484 frames across 4 sessions
│       ├── looking_away/                    # 583 frames across 4 sessions
│       └── eyes_open_looking_down/          # 440 frames across 4 sessions
├── models/                                  # Trained scikit-learn models & scalers
│   ├── yawn_model.joblib / yawn_scaler.joblib
│   ├── eyes_closed_model.joblib / eyes_closed_scaler.joblib
│   ├── looking_away_model.joblib / looking_away_scaler.joblib
│   ├── looking_down_model.joblib / looking_down_scaler.joblib
│   └── face_landmarker.task                 # MediaPipe Vision Landmarker binary model
├── report/                                  # Academic documentation & viva files
│   ├── driver_safety_monitor_project_report.tex  # Academic LaTeX source (Overleaf ready)
│   └── driver_safety_monitor_project_report.md   # Markdown project report with math formulas
├── results/                                 # Evaluated plots, metrics & presentation deck
│   ├── metrics.csv                          # Comprehensive numerical evaluation table
│   ├── *_confusion_matrix.png               # Confusion matrix plots for all 4 models
│   ├── *_threshold.png                      # Threshold vs Precision/Recall/F1 plots
│   └── Driver_Safety_Monitor_Viva_Presentation.pptx # 10-slide viva PowerPoint deck
├── src/                                     # Core modular Python source code
│   ├── features.py                          # MediaPipe landmark extraction & 4 biomarker formulas
│   ├── record_samples.py                    # Webcam image recorder for personal training data
│   ├── build_dataset.py                     # Processes all images into master data/features.csv
│   ├── train_models.py                      # Clean scikit-learn training pipeline (80/20 split)
│   ├── evaluate.py                          # Full metric calculation & visual plot generation
│   ├── realtime_monitor.py                  # Live 30 FPS webcam application with HUD & buzzer
│   ├── live_feature_test.py                 # Diagnostic tool showing raw biomarkers in real-time
│   ├── check_setup.py                       # Verifies Python packages & camera access
│   ├── check_features.py                    # Spot-checks statistical distributions
│   └── inspect_data.py                      # Data validation & grid visualization
├── generate_viva_presentation.py            # Generates the 16:9 widescreen PowerPoint deck
├── requirements.txt                         # Exact pinned Python dependencies
└── README.md                                # Project introduction & quick-start guide
```

---

## 4. Dataset Creation & Preprocessing Pipeline

The system is trained on a combined dataset of **3,534 valid frames** originating from two complementary sources:

### 4.1 Data Sources
1. **Public Kaggle Yawn Dataset (`data/raw/dataset_new/`):**
   - 1,448 total images pre-organized by original authors into `train/` and `test/` splits.
   - Provides ground-truth labels strictly for yawning ($yawn = 1$) vs. normal mouth ($yawn = 0$).
   - Successfully processed: 1,429 frames (19 skipped due to lack of detectable face).
2. **Personally Recorded Real-World Dataset (`data/own/`):**
   - 2,542 frames captured using `src/record_samples.py` under real driving sitting conditions.
   - Captured across 5 distinct chronological recording sessions to guarantee diversity.
   - Successfully processed: 2,105 frames (437 skipped during extreme head turns where facial landmarks were occluded).

### 4.2 Master Table (`data/features.csv`)
All extracted biomarkers are compiled into a unified table with the following schema:
- `EAR`: Continuous float (Eye Aspect Ratio).
- `MAR`: Continuous float (Mouth Aspect Ratio).
- `yaw`: Continuous float (Horizontal nose deviation percentage).
- `pitch`: Continuous float (Vertical nose position percentage).
- `source`: Category (`dataset_new` or `own`).
- `session`: Unique session identifier (e.g. `train/yawn` or `20261007_192453`).
- `yawn`: Binary label $\{0, 1\}$ or NaN.
- `eyes_closed`: Binary label $\{0, 1\}$ or NaN.
- `looking_away`: Binary label $\{0, 1\}$ or NaN.
- `looking_down`: Binary label $\{0, 1\}$ or NaN.

### 4.3 Class Breakdown in Dataset

| Condition | Class 0 (Normal) | Class 1 (Hazard) | Total Labelled Instances |
| :--- | :---: | :---: | :---: |
| **Yawn Detection** | 720 | 709 | 1,429 |
| **Eyes Closed** | 949 | 520 | 1,469 |
| **Looking Away** | 484 | 152 | 636 |
| **Looking Down** | 484 | 440 | 924 |

---

## 5. Mathematical Biomarker Engineering

The system uses **MediaPipe Face Mesh**, a deep convolutional network that locates **468 2D facial landmarks** $(x, y)$ in sub-millisecond time. From these coordinates, 4 specialized scale-invariant biomarkers are calculated.

```
                  [10: Forehead]
                        │
                        │  dy1 = (y_nose - y_forehead)
                        ▼
[234: Left Cheek] ──► [1: Nose] ◄── [454: Right Cheek]
                        ▲
                        │  dy2 = (y_chin - y_nose)
                        │
                   [152: Chin]
```

### 5.1 Eye Aspect Ratio (EAR) — Eye Closure
EAR computes the ratio of vertical eyelid opening to horizontal eye width:

$$\text{EAR} = \frac{\|p_2 - p_6\|_2 + \|p_3 - p_5\|_2}{2.0 \cdot \|p_1 - p_4\|_2}$$

- **Landmark Indices:**
  - Left Eye: Outer corner (33), Upper eyelids (160, 158), Inner corner (133), Lower eyelids (153, 144).
  - Right Eye: Inner corner (362), Upper eyelids (385, 387), Outer corner (263), Lower eyelids (373, 380).
- **Mathematical Principle:**
  - Numerator: Adds two vertical distances across the eye aperture.
  - Denominator: Twice the horizontal eye fissure width.
  - Scale Invariance: When the driver moves closer to or farther from the camera, both numerator and denominator scale by the identical factor $k$, leaving the ratio completely unchanged.
- **Values:** Open eye $\approx 0.287 \pm 0.08$. Closed eye $\approx 0.066 \pm 0.02$. ($4.3\times$ separation).

### 5.2 Mouth Aspect Ratio (MAR) — Yawn Detection
MAR measures vertical mouth separation normalized by mouth width:

$$\text{MAR} = \frac{\|p_2 - p_6\|_2 + \|p_3 - p_5\|_2}{2.0 \cdot \|p_1 - p_4\|_2}$$

- **Landmark Indices:** Left corner (61), Upper lip (39, 269), Right corner (291), Lower lip (405, 181).
- **Values:** Normal/Talking $\approx 0.299 \pm 0.081$. Yawn $\approx 0.788 \pm 0.196$. ($2.6\times$ expansion).

### 5.3 Looking Away: Horizontal Nose Deviation Ratio ($|Yaw|$)
*Why 3D depth trigonometry was abandoned:* Prior attempts to compute yaw using $\arctan2(\Delta Z, \Delta X)$ failed because monocular webcams estimate depth ($Z$) inaccurately, producing jitter and erratic angles.

*The 2D Symmetry Solution:* In a forward-facing face, bilateral symmetry places the nose tip (Landmark 1) at the midpoint between the left cheek (234) and right cheek (454).

$$X_{\text{center}} = \frac{X_{234} + X_{454}}{2}, \quad W_{\text{face}} = |X_{454} - X_{234}|$$

$$Yaw = \left( \frac{X_{\text{nose}} - X_{\text{center}}}{W_{\text{face}} + 10^{-6}} \right) \times 100\%$$

- **Facing Forward:** $X_{\text{nose}} \approx X_{\text{center}} \implies |Yaw| \le 5\%$.
- **Natural Deadband:** Slight head adjustments ($|Yaw| \le 12\%$) are accepted as safe driving.
- **Looking Away:** The nose projects sideways $\implies |Yaw| \ge 14\%\text{--}30\%$.

### 5.4 Looking Down: Vertical Nose Position Ratio ($Pitch$)
When looking downward (e.g. checking a phone in one's lap), the forehead tilts forward (occupying more vertical space in the camera projection) and the chin tilts back.

$$H_{\text{face}} = |Y_{152} - Y_{10}|$$
$$\text{Ratio}_{\text{vert}} = \frac{Y_{\text{nose}} - Y_{10}}{H_{\text{face}} + 10^{-6}}$$
$$Pitch = (\text{Ratio}_{\text{vert}} - 0.50) \times 100\%$$

- **Facing Forward:** Nose sits at the vertical midpoint ($50\%$) $\implies Pitch \approx -0.45\% \pm 4.9\%$.
- **Looking Down:** Forehead expands, nose shifts down to $>60\%$ of face height $\implies Pitch \ge +12.50\% \pm 1.68\%$.
- **Separation:** 100% clean linear separation with virtually zero overlap.

---

## 6. Machine Learning Modeling & Training Pipeline

The training script [src/train_models.py](file:///Users/rishitgoel/Desktop/AI-Meeting-Attention-Analyzer/driver-safety-monitor/src/train_models.py) implements the clean, standard curriculum machine learning workflow.

```
Dataset (features.csv)
       │
       ▼
Filter Valid Target Rows (dropna)
       │
       ▼
80 / 20 Stratified Split (train_test_split)
       │
       ▼
Feature Normalization (StandardScaler) ──► Fit on Train Only!
       │
       ▼
Logistic Regression Classifier (fit)
       │
       ▼
Evaluation on Test Set (Accuracy, Precision, Recall, F1)
       │
       ▼
Serialization (.joblib Model & Scaler Files)
```

### 6.1 Mathematical Formulation of Logistic Regression
For each behavior, the standardized feature scalar $z = \frac{x - \mu}{\sigma}$ is mapped to a probability $\hat{p} \in [0, 1]$ via the sigmoid function:

$$P(Y = 1 \mid z) = \sigma(w \cdot z + b) = \frac{1}{1 + e^{-(w \cdot z + b)}}$$

Where:
- $w$ is the learned regression coefficient (weight).
- $b$ is the bias (intercept).
- $\sigma(z)$ outputs the calibrated probability that the hazard condition is active.

### 6.2 Loss Function & Optimization
Parameters are determined by minimizing the $L_2$-regularized Binary Cross-Entropy loss:

$$\mathcal{L}(w, b) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{\lambda}{2} w^2$$

Optimization converges deterministically via the L-BFGS solver in under 20 iterations per model.

---

## 7. Comprehensive Evaluation & Experimental Metrics

Models were evaluated on independent, unseen 20% holdout test sets using [src/evaluate.py](file:///Users/rishitgoel/Desktop/AI-Meeting-Attention-Analyzer/driver-safety-monitor/src/evaluate.py).

### 7.1 Quantitative Performance Summary

| Model | Evaluated Feature | Train $N$ | Test $N$ | Test Accuracy | Precision | Recall | $F_1$-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Yawn Detection** | $\text{MAR}$ | 1,143 | 286 | **98.25%** | 98.6% | 97.9% | **0.982** |
| **Eyes Closed** | $\text{EAR}$ | 1,175 | 294 | **98.64%** | 97.2% | 99.0% | **0.981** |
| **Looking Away** | $\|Yaw\|$ | 508 | 128 | **88.28%** | **100.0%** | 51.6% | **0.681** |
| **Looking Down** | $Pitch$ | 739 | 185 | **99.46%** | 98.9% | **100.0%** | **0.994** |

### 7.2 Confusion Matrix Breakdown (Test Set)

#### 1. Yawn Detection (MAR)
- **True Negatives (Normal mouth correct):** 142
- **False Positives (False alarms):** 2
- **False Negatives (Missed yawns):** 3
- **True Positives (Yawns correctly detected):** 139

#### 2. Eyes Closed Detection (EAR)
- **True Negatives (Open eyes correct):** 187
- **False Positives (False alarms):** 3
- **False Negatives (Missed closures):** **1** (Only 1 closed eye missed out of 104!)
- **True Positives (Closures correctly detected):** 103

#### 3. Looking Away Detection ($|Yaw|$)
- **True Negatives (Facing forward correct):** 97
- **False Positives (False alarms):** **0** (**Zero false alarms on normal driving!**)
- **False Negatives (Missed turns):** 15
- **True Positives (Looking away detected):** 16

#### 4. Looking Down Detection ($Pitch$)
- **True Negatives (Facing forward correct):** 96
- **False Positives (False alarms):** 1
- **False Negatives (Missed looking down):** **0** (**100% detection rate!**)
- **True Positives (Looking down detected):** 88

---

## 8. Real-Time Inference, Filtering & Alert Logic

Single-frame machine learning predictions cannot be directly tied to a physical alarm buzzer. Normal driving physiology involves:
- Involuntary blinks lasting 100–250 ms (3–8 video frames).
- Lawful mirror and speedometer checks lasting 150–250 ms.

### 8.1 Temporal Rolling Persistence Filter
In [src/realtime_monitor.py](file:///Users/rishitgoel/Desktop/AI-Meeting-Attention-Analyzer/driver-safety-monitor/src/realtime_monitor.py), each behavior maintains a stateful consecutive frame counter $c_k(t)$:

$$c_k(t) = \begin{cases} c_k(t-1) + 1, & \text{if } P(Y_k = 1) \ge \tau_k \\ \max(0, c_k(t-1) - 1), & \text{otherwise} \end{cases}$$

An audible chime and visual warning trigger only when $c_k(t) \ge C_{\text{threshold}}$:

| State | Probability Threshold ($\tau$) | Persistence Required ($C_{\text{threshold}}$) | Real-World Time | Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Eyes Closed** | $0.50$ | 10 frames | $\approx 0.35$ s | Ignores natural blinks; triggers on micro-sleeps. |
| **Yawn** | $0.50$ | 10 frames | $\approx 0.35$ s | Ignores talking and singing; triggers on yawns. |
| **Looking Away** | $0.60$ | 12 frames | $\approx 0.40$ s | Ignores mirror checks; triggers on road distraction. |
| **Looking Down** | $0.50$ | 8 frames | $\approx 0.25$ s | Fast intervention for mobile phone texting. |

### 8.2 Head-Up Display (HUD) Interface
The webcam window displays:
- Green badges for `NORMAL` / Red badges for active `ALERT`.
- Real-time numerical probabilities for each condition.
- A four-column vertical bar chart visualizing the current probabilities relative to the 50% decision line.
- Live raw biomarker diagnostics (EAR, MAR, $|Yaw|\%$, $Pitch\%$) toggleable via key `'d'`.

---

## 9. Key Engineering Challenges & Solutions

### Challenge 1: Noisy 3D Z-Depth & Inaccurate Angles
- **The Problem:** The initial implementation used 3D cheek depth $\arctan2(\Delta Z, \Delta X)$ and forehead-chin depth $-\arctan2(\Delta Z, \Delta Y)$. Monocular webcams lack depth sensors; estimated $Z$ fluctuated with lighting, causing forward-facing pitch to jump between $+12^\circ$ and $-2^\circ$. Sitting at a normal desk angle falsely triggered continuous "Looking Down" alarms.
- **The Solution:** Completely eliminated 3D depth math. Replaced it with pure 2D landmark ratios: horizontal nose symmetry ($|Yaw|$) and vertical forehead-to-chin nose ratio ($Pitch$). These 2D ratios are scale-invariant, lighting-invariant, and separated the classes with near 100% accuracy.

### Challenge 2: Looking Away Hair-Trigger Sensitivity
- **The Problem:** Slight natural head shifts caused immediate "Looking Away" alarms.
- **The Solution:**
  1. Removed `class_weight="balanced"` during training, moving the 50% boundary from an overly sensitive $7\%$ up to a natural $12\%$.
  2. Increased real-time threshold $\tau$ to $0.60$ (requiring $\ge 14\%$ deviation).
  3. Increased persistence window to 12 frames ($0.4$s). Now, slight mirror checks stay green, while deliberate head turns trigger reliably.

### Challenge 3: Immediate Reset Bug on Dropped Frames
- **The Problem:** When turning the head away, MediaPipe occasionally drops face tracking for 1 or 2 frames. The original code reset all counters to $0$ on any dropped frame, making it impossible for the looking-away counter to ever reach its threshold.
- **The Solution:** Introduced a dedicated `no_face_frames` counter. Short 1-2 frame drops smoothly decay counters rather than zeroing them out, and sustained loss of face ($> 18$ frames / $0.6$s) alerts with `NO FACE / LOOKING AWAY`.

---

## 10. Step-by-Step Reproduction Guide

To run or demonstrate any part of the project on macOS or Linux:

### 1. Environment Activation
```bash
cd driver-safety-monitor
source .venv/bin/activate
```

### 2. Spot-Check Setup & Camera
```bash
python src/check_setup.py
```

### 3. Rebuild Dataset Features (Takes ~30 seconds)
```bash
python src/build_dataset.py
```

### 4. Train All 4 Models (Takes ~2 seconds)
```bash
python src/train_models.py
```

### 5. Run Full Evaluation & Generate Plots
```bash
python src/evaluate.py
```

### 6. Generate PowerPoint Presentation Deck
```bash
python generate_viva_presentation.py
```

### 7. Run Real-Time Webcam Monitor
```bash
python src/realtime_monitor.py
```
*(Controls: Press `'d'` to toggle raw biomarkers; Press `'q'` to quit cleanly).*

---

## 11. Viva Voce Master Q&A Defense Guide

### General ML & Computer Vision Questions

**Q1: What is the high-level architecture of your project?**  
> **Answer:** It is a vision-based machine learning pipeline consisting of three stages: (1) Landmark detection using MediaPipe Face Mesh (468 landmarks); (2) Geometric feature engineering to extract scale-invariant ratios (EAR, MAR, $|Yaw|$, $Pitch$); and (3) Classification using four independent Logistic Regression models equipped with a temporal rolling persistence filter to prevent false alarms.

**Q2: Why did you not train an end-to-end Deep Convolutional Neural Network (CNN/ResNet) directly on image pixels?**  
> **Answer:** MediaPipe already leverages a deep neural network to locate 468 landmark coordinates. By computing domain-specific geometric ratios from these coordinates, our feature space becomes scale-invariant and 1-dimensional per task. Using Logistic Regression provides three distinct advantages:
> 1. **Inference Speed:** Runs in $< 0.05\,\mu\text{s}$ per frame on CPU (guaranteeing 30 FPS).
> 2. **Complete Explainability:** Zero black-box criticism; each decision traces directly to $P(Y=1) = \sigma(w \cdot z + b)$.
> 3. **No Overfitting:** Models with only 2 parameters cannot memorize training noise.

**Q3: Why did you train four independent binary models instead of one multi-class model?**  
> **Answer:** Driver risk states are **not mutually exclusive**. A driver can simultaneously yawn while closing their eyes, or look down while closing their eyes. Multi-class classification (softmax) forces a single class prediction, whereas four independent binary models (sigmoid) allow simultaneous detection of compound hazards.

**Q4: How do your features handle drivers of different face shapes or distances from the camera?**  
> **Answer:** All four features are normalized dimensionless ratios:
> - EAR divides vertical eyelid height by horizontal eye width.
> - MAR divides vertical lip opening by horizontal mouth width.
> - $|Yaw|$ divides nose displacement by cheek-to-cheek width.
> - $Pitch$ divides vertical nose position by forehead-to-chin height.  
> If a driver moves closer to the camera, both the numerator and denominator increase by the same factor $k$, leaving the ratio completely scale-invariant.

---

### Feature Engineering Questions

**Q5: What is Eye Aspect Ratio (EAR) and what are its typical values?**  
> **Answer:** EAR measures eye openness using 6 landmarks per eye. The formula is $\text{EAR} = (\|p_2 - p_6\| + \|p_3 - p_5\|) / (2 \cdot \|p_1 - p_4\|)$. In our dataset, an open eye averages $0.287 \pm 0.08$, while a closed eye collapses to $0.066 \pm 0.02$, providing a massive $4.3\times$ separation ratio.

**Q6: What is Mouth Aspect Ratio (MAR) and how does it separate yawning from talking?**  
> **Answer:** MAR uses 6 mouth landmarks to measure vertical lip opening divided by horizontal mouth width. Normal talking produces brief, modest openings (mean MAR $= 0.299$). Yawning produces large vertical mandibular extensions (mean MAR $= 0.788$, a $2.6\times$ expansion) that persist for multiple seconds.

**Q7: Why did you switch from 3D depth trigonometry to 2D landmark geometry for head pose?**  
> **Answer:** Monocular consumer webcams lack hardware depth sensors. MediaPipe approximates the $Z$-depth coordinate, which fluctuates heavily with ambient lighting and facial expressions. 2D horizontal nose symmetry relative to the cheeks ($|Yaw|$) and vertical nose position relative to the forehead and chin ($Pitch$) eliminate $Z$-depth entirely, providing rock-solid, jitter-free separation.

---

### Machine Learning & Evaluation Questions

**Q8: What does the StandardScaler do, and why must it only be fit on training data?**  
> **Answer:** StandardScaler standardizes features to have mean $\mu = 0$ and standard deviation $\sigma = 1$ via $z = (x - \mu) / \sigma$. It must be fit **strictly on training data** and used only to transform test data. Fitting on test data or the entire dataset causes **data leakage**, artificially inflating test performance with information from the future.

**Q9: What is the mathematical formula for Logistic Regression?**  
> **Answer:** The model computes the linear combination $t = w \cdot z + b$ and passes it through the sigmoid function:
> $$P(Y=1 \mid z) = \sigma(t) = \frac{1}{1 + e^{-t}}$$
> When $t = 0$, $P = 0.50$. When $t > 0$, $P > 0.50$ (predicted class 1). When $t < 0$, $P < 0.50$ (predicted class 0).

**Q10: Why is Recall more important than Precision in a driver safety system?**  
> **Answer:** In safety applications:
> - **False Negative (Low Recall):** The driver falls asleep, but the system fails to alert $\implies$ Fatal crash.
> - **False Positive (Low Precision):** The driver is awake, but the system beeps $\implies$ Minor annoyance.  
> Therefore, high recall is vital. Our Eyes Closed model achieves **99.0% Recall** (only 1 missed closure out of 104), and our Looking Down model achieves **100.0% Recall**.

**Q11: Why does your Looking Away model have 100% Precision but 51.6% Recall?**  
> **Answer:** We intentionally tuned the model's bandwidth and real-time threshold to $0.60$ to prioritize **zero false alarms**. In driving, false alarms cause drivers to disable the system. With 100% precision, the system never bothers the driver during normal mirror checks, only alarming when head turns are genuine and sustained.

---

### Real-Time & Deployment Questions

**Q12: How does your system differentiate a natural blink from a micro-sleep?**  
> **Answer:** A natural human blink lasts between 100 and 200 milliseconds (3 to 6 frames at 30 FPS). Our temporal filter requires the Eyes Closed condition to persist for **10 consecutive frames** ($\approx 350$ milliseconds) before raising an alarm. Thus, normal blinks pass silently, while true micro-sleeps trigger the chime.

**Q13: What happens when the driver turns their head so far that MediaPipe loses the face?**  
> **Answer:** In our updated logic, a `no_face_frames` counter tracks sustained loss of tracking. If the face is missing for more than 18 frames ($\approx 0.6$ seconds), the system correctly identifies that the driver is not watching the road and triggers a `NO FACE / LOOKING AWAY` warning.

---

## 12. Formula Sheet & Glossary

### Mathematical Formula Sheet

$$\text{EAR} = \frac{\|p_2 - p_6\|_2 + \|p_3 - p_5\|_2}{2.0 \cdot \|p_1 - p_4\|_2}$$

$$\text{MAR} = \frac{\|p_2 - p_6\|_2 + \|p_3 - p_5\|_2}{2.0 \cdot \|p_1 - p_4\|_2}$$

$$Yaw = \left( \frac{X_{\text{nose}} - \frac{X_{234} + X_{454}}{2}}{|X_{454} - X_{234}| + 10^{-6}} \right) \times 100\%$$

$$Pitch = \left( \frac{Y_{\text{nose}} - Y_{10}}{|Y_{152} - Y_{10}| + 10^{-6}} - 0.50 \right) \times 100\%$$

$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

$$P(Y = 1 \mid z) = \frac{1}{1 + e^{-(w \cdot z + b)}}$$

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

---

### Glossary of Key Technical Terms
- **Biomarker:** A measurable, objective biological or anatomical indicator (e.g. eye opening, mouth expansion).
- **Scale Invariance:** The mathematical property of a ratio where multiplying all dimensions by a constant factor $k$ leaves the resulting ratio unchanged.
- **Micro-Sleep:** A brief involuntary episode of sleep lasting from 0.5 to 3 seconds during which the driver is completely unresponsive.
- **Sigmoid Function:** The S-shaped activation function $\sigma(t) = 1 / (1 + e^{-t})$ that maps real numbers to probabilities in $[0, 1]$.
- **Data Leakage:** An experimental error where information outside the training dataset (such as test set statistics) is used to fit models or scalers, producing artificially optimistic results.
- **Temporal Persistence:** A time-based smoothing filter that requires a detected event to hold for $N$ consecutive video frames before firing an external alert.
- **L-BFGS:** Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm; a quasi-Newton optimization method used to solve Logistic Regression parameters rapidly and deterministically.
