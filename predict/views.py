from django.shortcuts import render
from .ml_model import HousePricePredictor
from predict.ml_model2 import PostQualityPredictor

def predict_price(request):
    prediction = None
    if request.method == 'POST':
        try:
            size = float(request.POST.get('size'))
            bedrooms = int(request.POST.get('bedrooms'))
            age = float(request.POST.get('age'))
            
            predictor = HousePricePredictor()
            prediction = predictor.predict(size, bedrooms, age)
        except (ValueError, TypeError):
            prediction = "Invalid input"
    
    return render(request, 'predict.html', {'prediction': prediction})

def predict_quality(request):
    if request.method == 'POST':
        try:
            reputation = float(request.POST.get('reputation'))
            delta = float(request.POST.get('delta'))
            
            predictor = PostQualityPredictor()
            predictor.load_model('post_quality_model.joblib')
            prediction = predictor.predict(reputation, delta)
            return render(request, 'predict.html', {'prediction': prediction})
        except (ValueError, TypeError):
            prediction = "Invalid input"
    else:
        print("Not a post request")