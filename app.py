from sklearn.metrics import mean_squared_error
from flask import Flask, request, jsonify, render_template # type: ignore
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import pickle
import os

app = Flask(__name__)

# Cargar el modelo y los scalers
model = keras.models.load_model('house_price_2.h5', compile=False)
with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        data = request.form.to_dict()
        
        # Convertir los datos a un DataFrame
        input_data = {
            'area': [float(data['area'])],
            'bedrooms': [int(data['bedrooms'])],
            'bathrooms': [int(data['bathrooms'])],
            'stories': [int(data['stories'])],
            'mainroad': [1 if data['mainroad'] == 'yes' else 0],
            'guestroom': [1 if data['guestroom'] == 'yes' else 0],
            'basement': [1 if data['basement'] == 'yes' else 0],
            'hotwaterheating': [1 if data['hotwaterheating'] == 'yes' else 0],
            'airconditioning': [1 if data['airconditioning'] == 'yes' else 0],
            'parking': [int(data['parking'])],
            'prefarea': [1 if data['prefarea'] == 'yes' else 0],
            'furnishingstatus_furnished': [1 if data['furnishingstatus'] == 'furnished' else 0],
            'furnishingstatus_semi-furnished': [1 if data['furnishingstatus'] == 'semi-furnished' else 0],
            'furnishingstatus_unfurnished': [1 if data['furnishingstatus'] == 'unfurnished' else 0]
        }
        
        df = pd.DataFrame(input_data)
        
        # Crear características adicionales como en el notebook
        df['area_bedrooms'] = df['area'] * df['bedrooms']
        df['bathrooms_stories'] = df['bathrooms'] * df['stories']
        df['area_parking_ratio'] = df['area'] / (df['parking'] + 1)
        df['bedrooms_bathrooms_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)
        df['area_squared'] = df['area'] ** 2
        df['mainroad_prefarea'] = (df['mainroad'] & df['prefarea']).astype(int)
        df['hvac_status'] = df['hotwaterheating'] + 2 * df['airconditioning']
        df['bedrooms_bathrooms_stories'] = df['bedrooms'] * df['bathrooms'] * df['stories']
        df['area_per_bedroom'] = df['area'] / (df['bedrooms'] + 1)
        df['area_per_bathroom'] = df['area'] / (df['bathrooms'] + 1)
        df['bedrooms_squared'] = df['bedrooms'] ** 2
        df['bathrooms_squared'] = df['bathrooms'] ** 2
        df['stories_squared'] = df['stories'] ** 2
        
        # Seleccionar las mismas características que en el notebook
        features = ['area', 'bedrooms', 'bathrooms', 'stories', 'area_bedrooms', 
                   'bathrooms_stories', 'area_parking_ratio', 'bedrooms_bathrooms_ratio', 
                   'area_squared', 'bedrooms_bathrooms_stories', 'area_per_bedroom',
                   'area_per_bathroom', 'bedrooms_squared', 'bathrooms_squared', 
                   'stories_squared', 'hvac_status']
        
        X = df[features]
        
        # Escalar los datos
        X_scaled = scaler_X.transform(X)
        
        # Hacer la predicción
        y_pred_scaled = model.predict(X_scaled)
        
        # Deshacer la normalización y la transformación logarítmica
        y_pred_original_scaled = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_pred_original = np.expm1(y_pred_original_scaled)
        
        # Formatear el resultado
        predicted_price = round(float(y_pred_original[0][0]), 2)
        
        return jsonify({'predicted_price': predicted_price})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)