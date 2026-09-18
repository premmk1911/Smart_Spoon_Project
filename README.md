# 🥄 Smart Spoon — AI-Based Taste Enhancement & Low-Sodium Food Analysis

An AI-powered web application designed to explore **taste enhancement for low-sodium diets** by combining **food image classification** with **user dietary and taste-preference data**.

The system accepts a food image and a structured user survey, processes both sources of information, and generates predicted food-related parameters including **amplitude, frequency, and pulse width**. A manual food-class override is also available when required.

The application is implemented as a **Flask web application** with machine-learning models built using Python and scikit-learn.

---

## 🎯 Project Objective

People following low-sodium diets may reduce salt intake but can experience reduced taste satisfaction.

The Smart Spoon project explores an AI-assisted approach where information about:

* 🍛 Food type
* 🧂 Salt usage
* 🥗 Dietary habits
* 👤 User characteristics
* 😋 Taste preferences
* 📷 Food images

can be combined to estimate parameters that could be used as part of a future taste-enhancement system.

> **Note:** This repository represents an AI/ML prototype and does not directly control a physical smart spoon or provide medical advice.

---

# ✨ Key Features

### 📷 Food Image Classification

The application accepts an image of a food dish and uses an image-classification pipeline to identify the food category.

The current application defines these classes:

```text
dal
sambar
biryani
curry
snacks
roti
pickles
```

The detected food can also be manually overridden by the user when necessary.

---

### 📝 Dietary & Taste Survey

The application collects structured information about the user, including:

* Age
* Gender
* Low-sodium diet usage
* Dietary condition
* Dining frequency
* Satisfaction with low-sodium food
* Salt/condiment usage
* Awareness of taste-enhancement technology
* Interest in the device
* Importance of taste enhancement
* Purchase consideration
* Technology concerns
* Salt taste opinion
* Salt usage across different food categories

The Flask application converts these categorical survey responses into numerical features before passing them to the prediction pipeline.

---

### 🤖 Machine Learning Prediction

The project uses machine-learning models to estimate:

```text
Amplitude
Frequency
Pulse Width
```

The survey-only training pipeline uses a **Random Forest Regressor** with 100 estimators.

---

### 🌐 Flask Web Application

The project provides a Flask-based web interface.

The application:

1. Accepts a food image.
2. Saves the uploaded image.
3. Processes the survey responses.
4. Converts categorical responses into numerical values.
5. Sends the data to the prediction pipeline.
6. Identifies the food category.
7. Generates predicted parameters.
8. Displays the results.

The main Flask application is implemented in `app.py`.

---

# 🧠 System Architecture

```text
                  ┌───────────────────────┐
                  │       User            │
                  └───────────┬───────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
        ┌───────────────┐           ┌───────────────┐
        │  Food Image   │           │ User Survey   │
        └───────┬───────┘           └───────┬───────┘
                │                           │
                ▼                           ▼
        ┌───────────────┐           ┌───────────────┐
        │ Image Feature │           │ Survey Feature│
        │ Extraction    │           │ Processing    │
        └───────┬───────┘           └───────┬───────┘
                │                           │
                └─────────────┬─────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ Prediction       │
                    │ Pipeline         │
                    └────────┬─────────┘
                             │
                             ▼
              ┌─────────────────────────────┐
              │ Predicted Parameters        │
              │                             │
              │ • Amplitude                │
              │ • Frequency                │
              │ • Pulse Width              │
              └─────────────────────────────┘
```

---

# 🛠️ Tech Stack

| Technology           | Purpose                       |
| -------------------- | ----------------------------- |
| **Python**           | Core programming language     |
| **Flask**            | Web application framework     |
| **scikit-learn**     | Machine learning              |
| **Random Forest**    | Regression model              |
| **Joblib**           | Model serialization           |
| **Pandas**           | Data processing               |
| **NumPy**            | Numerical operations          |
| **HTML/CSS**         | Web interface                 |
| **Machine Learning** | Prediction and classification |

The repository contains separate scripts for survey processing, image features, model training, evaluation, and prediction.

---

# 📂 Project Structure

```text
Smart_Spoon_Project/
│
├── app/
│   └── app.py
│
├── scripts/
│   ├── data_loader.py
│   ├── evaluate_classifier.py
│   ├── image_features.py
│   ├── predict.py
│   ├── survey_features.py
│   ├── train_image_classifier.py
│   └── train_models.py
│
├── models/
│   ├── class_indices.pkl
│   ├── image_features.pkl
│   ├── survey_only_model.pkl
│   └── survey_pipeline.pkl
│
├── __init__.py
│
└── README.md
```

The current GitHub repository contains the model artifacts and the Python scripts used for feature processing, training, evaluation, and prediction.

---

# 🔄 How It Works

## Step 1 — User Uploads Food Image

The user uploads an image of a dish through the Flask interface.

The application securely handles the uploaded filename and stores the image in the application's upload directory.

---

## Step 2 — User Completes the Survey

The application collects information related to:

```text
Age
Gender
Diet
Health/Dietary Condition
Salt Usage
Taste Preference
Low-Sodium Satisfaction
Technology Awareness
Device Interest
Purchase Consideration
```

Categorical responses are mapped into numerical representations before prediction.

---

## Step 3 — Food Classification

The uploaded food image is passed through the image-processing/classification pipeline.

The currently supported food classes are:

```text
Dal
Sambar
Biryani
Curry
Snacks
Roti
Pickles
```

The project includes a dedicated image-classifier training script and serialized classification-related artifacts.

---

## Step 4 — Feature Processing

The survey and image information are transformed into features that can be consumed by the machine-learning pipeline.

The project separates:

```text
Survey Features
       +
Image Features
       ↓
Prediction Pipeline
```

---

## Step 5 — Prediction

The prediction module produces:

```text
Amplitude
Frequency
Pulse Width
```

The Flask application retrieves these values from the prediction result and displays them on the results page.

---

# 🤖 Machine Learning

## Survey-Based Model

The survey-only model uses:

**Algorithm:**

```text
Random Forest Regressor
```

**Configuration:**

```text
n_estimators = 100
random_state = 42
```

The model is trained using processed survey features and evaluated using the test-set R² score.

---

## Image Classification

The repository includes a dedicated training script for the food image classifier.

The classifier is associated with the food classes used by the application:

```text
dal
sambar
biryani
curry
snacks
roti
pickles
```

The project also stores class-index information required by the prediction pipeline.

---

# 📊 Prediction Output

After processing the input, the application displays:

| Output            | Description                                 |
| ----------------- | ------------------------------------------- |
| **Detected Dish** | Food category identified from the image     |
| **Used Dish**     | Detected category or user-selected override |
| **Amplitude**     | Predicted amplitude parameter               |
| **Frequency**     | Predicted frequency parameter               |
| **Pulse Width**   | Predicted pulse-width parameter             |

The results are passed from `predict()` to the Flask result template.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/premmk1911/Smart_Spoon_Project.git
```

```bash
cd Smart_Spoon_Project
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

Install the required Python packages:

```bash
pip install flask
pip install pandas
pip install numpy
pip install scikit-learn
pip install joblib
```

If additional packages are required by the image-processing implementation, install those according to the imports in the corresponding scripts.

---

# ▶️ Running the Application

From the project root:

```bash
python app/app.py
```

The Flask application starts in development mode.

Open the local URL shown in the terminal, typically:

```text
http://127.0.0.1:5000/
```

---

# 🧪 Model Training

The repository includes scripts for training the machine-learning components.

### Train Survey Model

```bash
python scripts/train_models.py
```

The training process:

1. Loads survey data.
2. Creates the target variables.
3. Splits the data into training and testing sets.
4. Applies the survey preprocessing pipeline.
5. Trains the Random Forest Regressor.
6. Calculates the test R² score.
7. Saves the trained model.

The resulting model is saved as:

```text
models/survey_only_model.pkl
```

---

# 🖼️ Image Model Training

The repository contains:

```text
train_image_classifier.py
```

for training the food image classification component.

The trained classification artifacts are stored in the project for use during prediction.

---

# 🔮 Prediction Pipeline

The central prediction flow is handled by:

```text
scripts/predict.py
```

The Flask application calls:

```python
result = predict(survey_answers, img_path)
```

and uses the returned values to generate the result page.

---

# 📌 Important Notes

### Prototype Status

This project is an **AI/ML prototype** exploring the relationship between food type, dietary preferences, salt usage, and taste-enhancement parameters.

The current repository does **not demonstrate direct hardware control of a physical spoon**.

### Medical Disclaimer

This project should not be considered a medical device or a substitute for professional dietary or medical advice.

The machine-learning outputs are experimental predictions intended for research and prototype development.

---

# 🚀 Future Enhancements

Possible future improvements include:

* [ ] Connect predictions to a physical smart-spoon prototype
* [ ] Real-time control of taste-enhancement hardware
* [ ] Expand food-image dataset
* [ ] Add more Indian food categories
* [ ] Improve image classification accuracy
* [ ] Collect larger real-world survey datasets
* [ ] Replace simulated targets with experimentally measured targets
* [ ] Add model-performance dashboards
* [ ] Add user accounts and prediction history
* [ ] Deploy the Flask application
* [ ] Add REST API support
* [ ] Add mobile application support
* [ ] Support additional regional languages

---

# 🔬 Research Direction

The project can be extended toward an end-to-end system:

```text
              User
                │
                ▼
        ┌───────────────┐
        │ Food Image    │
        └───────┬───────┘
                │
                ▼
        Food Classification
                │
                ▼
        ┌───────────────┐
        │ Food Category │
        └───────┬───────┘
                │
                │
User Survey ────┤
                │
                ▼
        Feature Engineering
                │
                ▼
          ML Prediction
                │
                ▼
     ┌─────────────────────┐
     │ Amplitude           │
     │ Frequency           │
     │ Pulse Width         │
     └──────────┬──────────┘
                │
                ▼
      Future Hardware Layer
                │
                ▼
        Smart Spoon Device
```

This would allow the current software prototype to become part of a larger AI-assisted hardware system.

---

# 👨‍💻 Author

**Prem Kumar M K**

GitHub:
https://github.com/premmk1911

Project Repository:
https://github.com/premmk1911/Smart_Spoon_Project

---

# 📄 License

No explicit license is currently specified in the repository.

If you intend to make the project open source, consider adding an appropriate license.
