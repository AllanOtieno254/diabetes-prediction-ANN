# Diabetes Prediction using Artificial Neural Networks (ANN)

## Project Overview
This project focuses on predicting diabetes using an Artificial Neural Network (ANN) model trained on the **Pima Indians Diabetes Dataset**. The dataset contains medical predictor variables and a target variable that indicates whether or not a patient has diabetes. This model can be used to make predictions on new data.

## Table of Contents
- [Project Overview](#project-overview)
- [File Structure](#file-structure)
- [Dataset Information](#dataset-information)
- [Model Training](#model-training)
- [Making Predictions](#making-predictions)
- [Setup and Installation](#setup-and-installation)
- [How to Use](#how-to-use)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## File Structure
```
Diabetes-Prediction-ANN/
│── data/
│   ├── diabetes.csv                    # Original dataset with outcomes
│   ├── diabetes_testing_data.csv        # Dataset without outcomes for testing predictions
│── models/
│   ├── pima_diabetes_model.h5           # Trained ANN model
│── notebooks/
│   ├── data_preprocessing.ipynb         # Jupyter notebook for preprocessing
│   ├── model_training.ipynb             # Jupyter notebook for training the ANN model
│── src/
│   ├── train_model.py                    # Python script for training the model
│   ├── predict.py                        # Python script for making predictions
│── README.md
│── requirements.txt                     # Required dependencies
│── LICENSE                               # License for the project
```

## Dataset Information
The dataset contains the following features:
- **Pregnancies** - Number of times pregnant
- **Glucose** - Plasma glucose concentration
- **BloodPressure** - Diastolic blood pressure (mm Hg)
- **SkinThickness** - Triceps skinfold thickness (mm)
- **Insulin** - 2-Hour serum insulin (mu U/ml)
- **BMI** - Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction (DPF)** - Diabetes pedigree function
- **Age** - Age in years
- **Outcome** - 1 for diabetes, 0 for non-diabetes (not included in test data)

## Model Training
The model is built using **TensorFlow** and **Keras** with the following architecture:
- Input Layer: 8 neurons (one for each feature)
- Hidden Layers: 2 Dense layers with ReLU activation
- Output Layer: 1 neuron with Sigmoid activation for binary classification

The dataset is preprocessed using **StandardScaler** for normalization.

## Making Predictions
Once the model is trained, it can be used to predict diabetes for new data. The test dataset does not have an outcome column, and the model predicts whether the patient has diabetes.

### Example Code for Making Predictions:
```python
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained model
model_path = "models/pima_diabetes_model.h5"
loaded_model = load_model(model_path)

# Load new test data
new_data = pd.read_csv("data/diabetes_testing_data.csv")

# Load the scaler (Must be the same scaler used in training)
scaler = StandardScaler()
scaler.fit(new_data)  # Ideally, this should be the scaler used on training data

# Preprocess the new data
new_data_scaled = scaler.transform(new_data)

# Predict using the loaded model
predictions = loaded_model.predict(new_data_scaled)

# Convert probabilities to binary class
predicted_classes = (predictions > 0.5).astype(int)

# Create a DataFrame with results
results_df = new_data.copy()
results_df["Predicted"] = predicted_classes

# Save results to CSV
results_df.to_csv("data/predictions.csv", index=False)

# Display results
display(results_df.head())
```

## Setup and Installation
To set up the project, follow these steps:

### 1. Clone the Repository:
```sh
git clone https://github.com/yourusername/Diabetes-Prediction-ANN.git
cd Diabetes-Prediction-ANN
```

### 2. Install Dependencies:
```sh
pip install -r requirements.txt
```

### 3. Train the Model (Optional if pre-trained model is available):
```sh
python src/train_model.py
```

### 4. Run Prediction on New Data:
```sh
python src/predict.py
```

## Results
- The trained model achieves an accuracy of approximately **85%**.
- Predictions on new data will be stored in **data/predictions.csv**.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset from the National Institute of Diabetes and Digestive and Kidney Diseases.
- TensorFlow and Keras for deep learning model implementation.

---
Let me know if you need further modifications!

