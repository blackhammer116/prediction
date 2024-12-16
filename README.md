# README.md
# Django ML House Price Predictor

## Project Overview
A simple Django web application demonstrating machine learning model integration for house price prediction.

## Setup Instructions
1. Clone the repository
2. Create a virtual environment


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

python -c "from predict.ml_model import HousePricePredictor; HousePricePredictor()"

python manage.py migrate

python manage.py runserver


# Project Structure:
# prediction/
#   ├── manage.py
#   ├── requirements.txt
#   ├── ml_house_prediction/
#   │   ├── __init__.py
#   │   ├── settings.py
#   │   ├── urls.py
#   │   └── wsgi.py
#   ├── predict/
#   │   ├── __init__.py
#   │   ├── admin.py
#   │   ├── apps.py
#   │   ├── models.py
#   │   ├── views.py
#   │   ├── ml_model.py
#   │   └── templates/
#   │       └── predict.html
#   └── ml_model.joblib