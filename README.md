# Credit Card Fraud Detection

This project implements a machine learning pipeline to detect fraudulent credit card transactions. It includes scripts for preprocessing data, training the model, testing, and a basic deployment simulation.

## Project Structure
```
.
├── .vscode/ # VS Code settings (optional)
├── pycache/ # Python bytecode cache
├── models/ # Directory to save trained models
├── creditcard.csv # Dataset (CSV format)
├── creditcard.xlsx # Dataset (Excel format)
├── deployment.py # Script to load and run the trained model on new data
├── fraud_detection.py # Core logic for fraud detection
├── preprocess.py # Data preprocessing logic
├── test.py # Script to test model performance
├── train_model.py # Model training script
├── .gitignore # Git ignore configuration
└── README.md # Project documentation
```

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
- joblib (for model saving/loading)

Install dependencies using pip:
```
pip install -r requirements.txt
```
## How to Run

1. **Preprocess the data**
```
python preprocess.py
```

This will clean and prepare the `creditcard.csv` dataset for model training.

2. **Train the model**
```
python train_model.py
```
This will train a machine learning model and save it to the `models/` directory.

3. **Test the model**
```
python test.py
```
This will evaluate the trained model's performance on the test set.

4. **Deploy and predict**
```
python deployment.py
```
This simulates a deployment scenario and makes predictions using the saved model.

## Notes

- Make sure the `creditcard.csv` file is in the root directory.
- All intermediate files (preprocessed data, model artifacts) are saved automatically.
- If you want to use a different dataset, ensure it follows a similar format (features + 'Class' label).
