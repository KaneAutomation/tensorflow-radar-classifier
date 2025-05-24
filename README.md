# TensorFlow Radar Signal Classifier

This project builds a binary classifier to identify whether ionosphere radar signals are "good" (indicating structure) or "bad" (no structure) using a neural network built with TensorFlow/Keras. It was completed as part of a machine learning course at Purdue University Global.

## 📊 Dataset
- **Source**: [UCI Ionosphere Dataset](https://archive.ics.uci.edu/ml/datasets/Ionosphere)
- **Samples**: 351 radar returns
- **Features**: 34 numeric signal values
- **Label**: "g" (good) or "b" (bad)

## 🧠 Model
- Built using `tf.keras.Sequential`
- 2 hidden layers with ReLU activations
- Output layer with sigmoid activation
- Binary crossentropy loss, Adam optimizer

## 🔍 Performance
Evaluated on a held-out test set:

| Metric      | Score     |
|-------------|-----------|
| Accuracy    | 95.45%    |
| Precision   | 94.83%    |
| Recall      | 98.21%    |
| F1 Score    | 96.49%    |

## 💡 Key Learnings
- Data preprocessing (label encoding, scaling)
- Train/test splits with stratification
- Binary neural network classification
- Evaluating performance with scikit-learn

## 🔗 Technologies Used
- Python 3.12
- TensorFlow / Keras
- Scikit-learn
- VS Code + GitHub

## 📁 Files
- `main.py` – full training and evaluation code
- `assignment_writeup.docx` – summary and analysis for class submission

---

### ✨ How to Run
1. Clone the repo
2. Set up a virtual environment and install dependencies:
   ```bash
   pip install tensorflow scikit-learn pandas
   ```
3. Run the script:
   ```bash
   python main.py
   ```
## 🔖 Certification

This project was completed as part of the **Data Intelligence Micro-Credential** issued by Purdue University Global.  
🎓 [View Verified Credential](https://api.badgr.io/public/assertions/SoeCYMQwQtyYtAMRCpvmbQ?identity_email=ck106194%40hotmail.com)

