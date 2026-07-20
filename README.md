
Customer Analytics & Marketing Optimization Pipeline

An end-to-end customer analytics framework combining data engineering, dimensionality reduction, deep supervised learning, and reinforcement learning policy evaluation.
📁 Repository Structure
Plaintext

project-root/
├── data/                             # Raw and processed CSV data stores
├── notebooks/                        # Development and modeling notebooks
│   ├── 01_data_preprocessing.ipynb   # Raw cleaning and feature engineering
│   ├── 02_pca_lda.ipynb               # Linear & non-linear dimensionality reductions
│   ├── 03_classification.ipynb       # MLP neural network classifier for high-value labeling
│   ├── 04_regression.ipynb           # MLP neural network regressor for monetary forecasting
│   └── 05_qlearning_dqn.ipynb       # Tabular Q-Learning vs Deep Q-Networks (DQN)
├── src/                              # Reusable pipeline .py source code modules
├── models/                           # Serialized .pkl and .pt model artifacts
├── app.py                            # Streamlit GUI interactive application entry point
├── category_map.json                 # Keyword-to-category mapping rules
├── requirements.txt                  # Strict environment dependency listings
├── README.md                         # Project setup and execution documentation
└── report.pdf                        # Final formal written project report

⚙️ Environment Setup & Execution

1. Initialize Virtual Environment
   Bash

python -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

2. Install Dependencies
   Bash

pip install -r requirements.txt

3. Launch the Dashboard
   Bash

streamlit run app.py
