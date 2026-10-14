
# Cost Function Behaviour in Linear Regression

A Django-based machine learning project that demonstrates how Gradient Descent minimizes the cost function in Linear Regression.

## Features

- Upload and preview CSV datasets
- Train a custom Linear Regression model using Gradient Descent
- Visualize cost reduction across training iterations
- Compare different input feature combinations
- Compare learning rates: 0.001, 0.01, and 0.1
- Evaluate models using MAE, MSE, RMSE, and R-squared
- Display experiment results and cost curves

## Technology Stack

- Python
- Django
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- HTML, CSS, Bootstrap

## Project Structure

- `analyzer/` — Django application and views
- `config/` — Django project configuration
- `ml_engine/` — Custom Linear Regression implementation
- `templates/analyzer/` — Dashboard and base templates
- `manage.py` — Django management utility
- `requirements.txt` — Python dependencies
- `student_performance_sample.csv` — Sample dataset, if included

## Setup Instructions

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies.
4. Run Django migrations.
5. Start the development server.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/ in your browser.

## Project Purpose

This project provides a practical way to understand cost function behaviour, Gradient Descent, learning-rate selection, and regression model evaluation.

## Author

MCA Student
