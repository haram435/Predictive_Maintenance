# 🛠️ Predictive Maintenance — Machine Failure Type Classifier

A Streamlit-deployed ML app that predicts *what kind* of failure an industrial machine is heading toward — not just whether it will fail — from live sensor readings.

🔗 **Live Demo:** https://predictivemaintenance-mpjynhp3rdkcmvrwpntohz.streamlit.app/

---

## Overview

Given a machine's type, air/process temperature, rotational speed, torque, and tool wear, the model classifies it into one of six states:
`No Failure`, `Heat Dissipation Failure`, `Power Failure`, `Overstrain Failure`, `Tool Wear Failure`, `Random Failures`.

This is a 6-class classification problem, not simple binary failure detection.

## Dataset

- ~10,000 machine readings (AI4I 2020-style predictive maintenance dataset)
- Severely imbalanced: 9,652 "No Failure" rows vs. 348 failures spread across 5 failure types — as few as **18 total examples** for the rarest class (Random Failures)

## Approach

- **Feature engineering:** added `Temp_difference` (Process − Air temperature) and `Power_Index` (Torque × Rotational speed), based on patterns found during EDA
- **Preprocessing:** `StandardScaler` on numeric features, `OneHotEncoder` on machine `Type`, wrapped in a single `sklearn` Pipeline
- **Compared two models**, both with `class_weight='balanced'` to counter the imbalance:

| Model | Balanced Accuracy | Raw Accuracy | Behavior |
|---|---|---|---|
| Random Forest | 0.57 | **0.99** | Near-perfect overall, but 0% recall on the two rarest failure types — defaults to "No Failure" when a class has too few examples to learn |
| **Logistic Regression (chosen)** | **0.74** | 0.64 | Lower raw accuracy, but actively catches rare failure types at the cost of more false alarms |

**Why Logistic Regression:** in a maintenance context, missing a real failure is more costly than a false alarm, so a model that tries on rare classes is more useful in practice than one that's silently blind to them — even though it scores lower on paper.

## Key Finding

Random Failures has only 18 total rows in the entire dataset (14 in training). No algorithm can reliably learn a class from that few examples — this is a **data limitation, not a modeling bug**. Manually testing the deployed app confirmed the same pattern the confusion matrix showed: healthy machines are occasionally misclassified as "Random Failures," and Tool Wear inputs are sometimes confused with Overstrain Failure — both consistent with how rare those classes are in training.

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · Streamlit · Supabase (prediction logging)

## Run Locally

```bash
git clone https://github.com/haram435/Predictive_Maintenance.git
cd Predictive_Maintenance
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements

- Group the rarest failure types into a single "Other Failure" bucket for more stable metrics
- Stratified k-fold cross-validation instead of a single train/test split
- Per-class threshold tuning rather than relying on one aggregate metric
