# 🌿 Análisis de Cenicilla Mendely

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 📘 Descripción General

El proyecto **Análisis de Cenicilla Mendely** es una aplicación basada en **visión por computadora** e **inteligencia artificial** que permite analizar hojas de mango y detectar la presencia de **Cenicilla**, una enfermedad común que afecta la productividad del cultivo.  

El sistema integra distintos **modelos de Machine Learning y Deep Learning** (SVM, CNN, Vision Transformer, EfficientNet, ConvNeXt y Swin Transformer) entrenados con imágenes reales de hojas de mango, proporcionando una clasificación precisa de la severidad de la enfermedad mediante el reconocimiento visual.

El conjunto de datos utilizado para el entrenamiento está conformado por **600 imágenes etiquetadas por cada una de las 5 clases de severidad**, totalizando **3.000 imágenes**.

---

## 🧠 Modelos de Clasificación

El proyecto incluye seis modelos de entrenamiento y análisis, de complejidad creciente:

- **SVM (Support Vector Machine)** — sobre 35 características extraídas manualmente (color, textura, bordes)
- **CNN (Convolutional Neural Network)** — arquitectura propia estilo VGG
- **Vision Transformer (ViT)** — implementación personalizada basada en parches
- **EfficientNet (EfficientNetB0)** — entrenado desde cero
- **ConvNeXt (ConvNeXtTiny)** — entrenado desde cero
- **Swin Transformer** — implementación personalizada con ventanas de atención

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

Estas métricas permiten comparar los seis modelos y determinar cuál tiene el mejor desempeño frente a las distintas clases de la enfermedad.

---

## 🖥️ Interfaz Gráfica

El sistema cuenta con una interfaz desarrollada en **Tkinter**, que permite:

- Cargar imágenes desde el dispositivo  
- Visualizar los resultados del análisis en tiempo real, por modelo  
- Mostrar la clase detectada junto con la imagen procesada  
- Ejecutar predicciones con los seis modelos entrenados  
- Comparar el desempeño de los seis modelos entre sí  

---

## ⚙️ Tecnologías Usadas

- 🐍 **Python 3.8+**  
- 🧩 **TensorFlow / Keras** – para redes neuronales (CNN, Vision Transformer, EfficientNet, ConvNeXt, Swin Transformer)  
- 🧠 **Scikit-learn** – para el modelo SVM  
- 🖼️ **OpenCV y Pillow (PIL)** – para procesamiento de imágenes  
- 📊 **NumPy / Matplotlib** – para visualización y cálculos  
- 💾 **joblib / h5py** – para manejo de modelos entrenados  
- 🪟 **Tkinter** – interfaz gráfica moderna y responsiva  

---

## 👨‍💻 Autor

**Luis Andrés Rodríguez Valle**  
Desarrollador del sistema de análisis de Cenicilla  
🌐 GitHub: [@MundoCode777](https://github.com/MundoCode777)