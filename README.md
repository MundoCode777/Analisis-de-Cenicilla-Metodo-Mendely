# 🌿 Análisis de Cenicilla Mendely

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 📘 Descripción General

El proyecto **Análisis de Cenicilla Mendely** es una aplicación basada en **visión por computadora** e **inteligencia artificial** que permite analizar hojas de mango y detectar la presencia de **Cenicilla**, una enfermedad común que afecta la productividad del cultivo.  

El sistema integra distintos **modelos de Machine Learning y Deep Learning** (SVM, CNN, Transformer) entrenados con imágenes reales de hojas de mango, proporcionando una clasificación precisa de las enfermedades mediante el reconocimiento visual.

---

## 🧠 Modelos de Clasificación

El proyecto incluye varios modelos de entrenamiento y análisis:

- **SVM (Support Vector Machine)**
- **CNN (Convolutional Neural Network)**
- **Transformer**

Cada modelo genera una **imagen representativa de los resultados**, mostrando las **5 clases de la enfermedad Cenicilla** identificadas en el conjunto de datos:

1. Clase 1 – Resistente  
2. Clase 2 – Moderadamente tolerante  
3. Clase 3 – Ligeramente tolerante  
4. Clase 4 – Susceptible  
5. Clase 5 – Altamente susceptible  

---

## 📊 Métricas Avanzadas

Cada modelo proporciona métricas avanzadas para evaluar su rendimiento:

- ✅ **Accuracy (Precisión general)**  
- 📈 **Precision y Recall (por clase)**  
- 🧮 **F1-Score**  
- 🔍 **Matriz de Confusión**  
- 📉 **Curva ROC / AUC**  
- 📊 **Reporte de Clasificación Completo**

Estas métricas permiten comparar los modelos y determinar cuál tiene el mejor desempeño frente a las distintas clases de la enfermedad.

---

## 🖥️ Interfaz Gráfica

El sistema cuenta con una interfaz desarrollada en **Tkinter**, que permite:

- Cargar imágenes desde el dispositivo  
- Visualizar los resultados del análisis en tiempo real  
- Mostrar la clase detectada junto con la imagen procesada  
- Ejecutar predicciones con los modelos entrenados  

---

## ⚙️ Tecnologías Usadas

- 🐍 **Python 3.8+**  
- 🧩 **TensorFlow / Keras** – para redes neuronales (CNN, Transformer)  
- 🧠 **Scikit-learn** – para modelos SVM  
- 🖼️ **OpenCV y Pillow (PIL)** – para procesamiento de imágenes  
- 📊 **NumPy / Matplotlib** – para visualización y cálculos  
- 💾 **joblib / h5py** – para manejo de modelos entrenados  
- 🪟 **Tkinter** – interfaz gráfica moderna y responsiva  

---

## 👨‍💻 Autor

**Luis Andrés Rodríguez Valle**  
Desarrollador del sistema de análisis de Cenicilla  
🌐 GitHub: [@MundoCode777](https://github.com/MundoCode777)
