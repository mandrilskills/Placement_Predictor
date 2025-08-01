


# 🎓 Placement Predictor

[![Streamlit App](https://img.shields.io/badge/Live_App-Click_to_Open-4CAF50?logo=streamlit\&logoColor=white)](https://placementpredictormsircar.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning web application that predicts whether a student will be placed or not based on academic and skill-based attributes. Built using **Streamlit** for frontend deployment and **Logistic Regression** for the prediction model.

---

## 🚀 Live Demo

🔗 [**Click here to try the Placement Predictor**](https://placementpredictormsircar.streamlit.app/)

---

## 📌 Features

* Predicts student placement outcome using academic and skill-related features
* User-friendly UI powered by Streamlit
* Clean visualizations and model output
* Simple and fast predictions
* Lightweight deployment

---

## 🧠 Model Overview

* **Algorithm Used**: Logistic Regression
* **Dataset**: Custom dataset containing features like CGPA, communication skills, etc.
* **Libraries**:

  * `pandas`
  * `scikit-learn`
  * `streamlit`
  * `matplotlib`

---

## 📁 Project Structure

```
Placement_Predictor/
├── placement.csv               # Dataset
├── Placement_Predictor.ipynb   # Jupyter notebook with EDA & model
├── placement.py                # Streamlit app
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ How to Run Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/mandrilskills/Placement_Predictor.git
   cd Placement_Predictor
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run placement.py
   ```

---

## 📊 Input Features

| Feature         | Description                    |
| --------------- | ------------------------------ |
| `cgpa`          | Cumulative Grade Point Average |
| `iq`            | Intelligence Quotient          |
| `communication` | Communication Skill Level      |

---

## ✅ Prediction Output

* **Placed** or **Not Placed** (binary classification)

---

## 📌 Future Improvements

* Add more student attributes (e.g., internships, projects)
* Model comparison (Random Forest, SVM, etc.)
* Improve dataset size and diversity
* Add feedback loop and confidence score

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
Feel free to open a pull request or raise an issue.

---

## 📬 Contact

Created by [Mandril Skills](https://github.com/mandrilskills)
🔗 Live App: [placementpredictormsircar.streamlit.app](https://placementpredictormsircar.streamlit.app/)

---

Would you like this README converted into a stylized **PDF** or Markdown **badge-rich** version for GitHub display?
