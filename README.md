# 🏡 Predicción de Precios de Viviendas

Este repositorio contiene un modelo de red neuronal 🤖 diseñado para predecir los precios de viviendas. Se ha desarrollado una API utilizando Flask 🐍 para servir el modelo, y se ha implementado una interfaz web en HTML 🌐 para facilitar la interacción con el usuario.

## 📂 Estructura del Proyecto

- 📄 `model`: Contiene el archivo del modelo de red neuronal entrenado (`house_price_2.h5`).
- 📄 `scalers`: Contiene los archivos de normalización (`scaler_X.pkl` y `scaler_y.pkl`).
- 📄 `api`: Contiene el código de la API desarrollada con Flask.
- 📁 `templates/`: Contiene los archivos HTML para la interfaz web.
- 📄 `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.

## 🛠 Tecnologías Utilizadas

- 🐍 **Python**: Lenguaje principal del proyecto.
- 🔬 **TensorFlow/Keras**: Para la construcción y entrenamiento de la red neuronal.
- 🌍 **Flask**: Para la creación del servicio API.
- 🎨 **HTML, CSS, JavaScript**: Para la interfaz de usuario.

## 🚀 Instalación y Uso

1. Clonar el repositorio:
   ```sh
   git clone https://github.com/anaBorja/Housing_Price.git
   cd Housing_Price
   ```

2. Instalar dependencias:
   ```sh
   pip install -r requirements.txt
   ```

3. Ejecutar la API:
   ```sh
   python app.py
   ```

4. Abrir el navegador 🌐 y acceder a la interfaz web en `http://localhost:5000`.

## ⚙️ Funcionamiento del Modelo

El modelo de red neuronal ha sido entrenado con un conjunto de datos de precios de viviendas, utilizando técnicas de normalización y optimización para mejorar la precisión de las predicciones. La API recibe datos de entrada en formato JSON, los procesa con el modelo y devuelve una predicción del precio de la vivienda.

## 🤝 Contribuciones

Se aceptan contribuciones para mejorar el modelo, la API y la interfaz web. Si deseas colaborar, por favor abre un issue o envía un pull request.

## 📜 Licencia

Este proyecto está bajo la licencia MIT.

