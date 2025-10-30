import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json

class CNNModel:
    def __init__(self):
        self.model = None
        self.img_size = (150, 150)
        self.class_names = {
            0: "Clase 0 - Resistente",
            1: "Clase 1 - Moderadamente tolerante", 
            2: "Clase 2 - Ligeramente tolerante",
            3: "Clase 3 - Susceptible",
            4: "Clase 4 - Altamente susceptible"
        }
        self.is_trained = False
        self.load_or_create_model()
    
    def load_labels(self):
        """Cargar etiquetas desde data/labels.json"""
        labels_file = "data/labels.json"
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r') as f:
                    labels = json.load(f)
                    print(f"✅ CNN: Cargadas {len(labels)} etiquetas desde {labels_file}")
                    return labels
            except Exception as e:
                print(f"⚠️ Error cargando etiquetas: {e}")
                return {}
        return {}
    
    def load_or_create_model(self):
        """Cargar modelo existente o crear uno nuevo"""
        model_path = "modelos/cnn_model.h5"
        
        # Verificar si hay etiquetas disponibles
        labels = self.load_labels()
        has_labels = len(labels) >= 10
        
        # Si hay etiquetas, ofrecer reentrenamiento
        if has_labels:
            if os.path.exists(model_path):
                print(f"\n📊 CNN: Modelo existente encontrado")
                print(f"📊 CNN: Tienes {len(labels)} imágenes etiquetadas")
                print(f"\n¿Qué deseas hacer?")
                print(f"1. Cargar modelo existente")
                print(f"2. Reentrenar con nuevas etiquetas")
                
                try:
                    choice = input("Selecciona (1 o 2): ").strip()
                    if choice == "2":
                        self.create_and_train_model(labels)
                        return
                except:
                    pass
            else:
                print(f"\n🎓 CNN: Entrenando modelo por primera vez con {len(labels)} imágenes...")
                self.create_and_train_model(labels)
                return
        
        # Intentar cargar modelo existente
        try:
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                self.is_trained = True
                print("✅ Modelo CNN cargado exitosamente")
            else:
                print("\n❌ CNN: No hay modelo entrenado ni etiquetas disponibles")
                print("📝 Para entrenar el modelo CNN:")
                print("   1. Ejecuta: python etiquetador.py")
                print("   2. Etiqueta al menos 50 imágenes")
                print("   3. Vuelve a ejecutar la aplicación")
                self.is_trained = False
        except Exception as e:
            print(f"⚠️ Error cargando modelo CNN: {e}")
            self.is_trained = False
    
    def create_model_architecture(self):
        """Crear arquitectura CNN mejorada"""
        model = keras.Sequential([
            # Bloque 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=(150, 150, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Bloque 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Bloque 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Bloque 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),
            
            # Capa densa
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Salida
            layers.Dense(5, activation='softmax')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_images_from_labels(self, labels):
        """Cargar imágenes desde las etiquetas"""
        X = []
        y = []
        valid_images = []
        failed_images = []
        
        print(f"📸 CNN: Cargando {len(labels)} imágenes...")
        
        for idx, (image_name, class_id) in enumerate(labels.items(), 1):
            image_path = os.path.join("data", image_name)
            
            if not os.path.exists(image_path):
                failed_images.append((image_name, "Archivo no encontrado"))
                continue
            
            try:
                # Cargar y preprocesar imagen
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    
                    X.append(img_array)
                    y.append(class_id)
                    valid_images.append(image_name)
            except Exception as e:
                failed_images.append((image_name, f"Error: {e}"))
            
            # Mostrar progreso cada 50 imágenes
            if idx % 50 == 0:
                print(f"   Cargadas: {idx}/{len(labels)}")
        
        if len(X) < 10:
            print(f"\n❌ CNN ERROR: Solo se cargaron {len(X)} imágenes válidas")
            if failed_images:
                print(f"⚠️ Imágenes con problemas:")
                for img, reason in failed_images[:5]:
                    print(f"   - {img}: {reason}")
            return None, None, valid_images
        
        return np.array(X), np.array(y), valid_images
    
    def create_and_train_model(self, labels=None):
        """Crear y entrenar modelo CNN con imágenes reales"""
        if labels is None:
            labels = self.load_labels()
        
        if not labels or len(labels) < 10:
            print("❌ CNN: Necesitas al menos 10 imágenes etiquetadas")
            self.is_trained = False
            return False
        
        print(f"\n{'='*60}")
        print(f"🔄 CNN: ENTRENANDO MODELO CON IMÁGENES REALES")
        print(f"{'='*60}\n")
        
        # Cargar imágenes
        X, y, valid_images = self.load_images_from_labels(labels)
        
        if X is None or len(X) < 10:
            self.is_trained = False
            return False
        
        print(f"\n✅ CNN: Datos preparados:")
        print(f"   - Imágenes válidas: {len(X)}")
        print(f"   - Shape: {X.shape}")
        print(f"   - Clases únicas: {np.unique(y)}")
        
        # Mostrar distribución de clases
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n📊 Distribución de clases:")
        for class_id, count in zip(unique, counts):
            percentage = (count / len(y)) * 100
            class_name = self.class_names[class_id]
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        # Verificar si se puede estratificar
        min_samples = min(counts)
        can_stratify = min_samples >= 2
        
        # Dividir en train/validation
        print(f"\n🔀 Dividiendo datos (80% entrenamiento, 20% validación)...")
        try:
            if can_stratify:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                print("   ⚠️ Algunas clases tienen pocas muestras, división sin estratificación")
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        except Exception as e:
            print(f"   ❌ Error en división: {e}")
            self.is_trained = False
            return False
        
        print(f"   - Train: {len(X_train)} imágenes")
        print(f"   - Validation: {len(X_val)} imágenes")
        
        # Data Augmentation para mejorar generalización
        print(f"\n🔄 Configurando Data Augmentation...")
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Crear modelo
        print(f"\n🏗️ Construyendo arquitectura CNN...")
        self.model = self.create_model_architecture()
        
        # Mostrar resumen del modelo
        print(f"\n📋 Resumen del modelo:")
        total_params = self.model.count_params()
        print(f"   Total parámetros: {total_params:,}")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Entrenar modelo
        print(f"\n🎓 Entrenando modelo CNN...")
        print(f"   - Épocas: 50 (con early stopping)")
        print(f"   - Batch size: 16")
        print(f"   - Data augmentation: Activado")
        print(f"\n{'='*60}\n")
        
        try:
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=16),
                validation_data=(X_val, y_val),
                epochs=50,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluar modelo
            print(f"\n{'='*60}")
            print(f"✅ CNN: RESULTADOS DEL ENTRENAMIENTO")
            print(f"{'='*60}")
            
            train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
            val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
            
            print(f"📊 Precisión en entrenamiento: {train_acc:.2%}")
            print(f"📊 Precisión en validación: {val_acc:.2%}")
            print(f"📉 Loss en entrenamiento: {train_loss:.4f}")
            print(f"📉 Loss en validación: {val_loss:.4f}")
            
            # Guardar modelo
            print(f"\n💾 Guardando modelo...")
            os.makedirs("modelos", exist_ok=True)
            self.model.save("modelos/cnn_model.h5")
            
            # Guardar metadata
            metadata = {
                'n_samples': len(X),
                'n_train': len(X_train),
                'n_val': len(X_val),
                'train_accuracy': float(train_acc),
                'val_accuracy': float(val_acc),
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'epochs_trained': len(history.history['loss']),
                'img_size': self.img_size,
                'classes': unique.tolist()
            }
            
            with open("modelos/cnn_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Modelo guardado en: modelos/cnn_model.h5")
            print(f"✅ Metadata guardada en: modelos/cnn_metadata.json")
            
            self.is_trained = True
            
            print(f"\n{'='*60}")
            print(f"🎉 ¡CNN ENTRENADA EXITOSAMENTE!")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error durante el entrenamiento: {e}")
            self.is_trained = False
            return False
    
    def preprocess_image(self, image_path):
        """Preprocesar imagen para predicción"""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = img.resize(self.img_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
                return np.expand_dims(img_array, axis=0)
        except Exception as e:
            print(f"⚠️ Error preprocesando imagen: {e}")
            return None
    
    def predict_image(self, image_path):
        """Predecir clase de una imagen"""
        if not self.is_trained or self.model is None:
            return {
                'class': -1,
                'class_name': "Modelo CNN no entrenado. Ejecuta etiquetador.py",
                'confidence': 0.0,
                'model': 'CNN'
            }
        
        try:
            processed_img = self.preprocess_image(image_path)
            
            if processed_img is None:
                return {
                    'class': -1,
                    'class_name': "Error al procesar imagen",
                    'confidence': 0.0,
                    'model': 'CNN'
                }
            
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            return {
                'class': int(predicted_class),
                'class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'probabilities': {
                    self.class_names[i]: float(predictions[0][i])
                    for i in range(len(predictions[0]))
                },
                'model': 'CNN'
            }
            
        except Exception as e:
            print(f"❌ Error en predicción CNN: {e}")
            return {
                'class': -1,
                'class_name': f"Error: {str(e)}",
                'confidence': 0.0,
                'model': 'CNN'
            }
    
    def analyze_dataset(self, data_folder="data"):
        """Analizar todas las imágenes en una carpeta"""
        if not self.is_trained:
            return [{
                'image_name': 'N/A',
                'class': -1,
                'class_name': 'Modelo CNN no entrenado',
                'confidence': 0.0,
                'model': 'CNN'
            }]
        
        results = []
        
        if not os.path.exists(data_folder):
            return [{
                'image_name': 'N/A',
                'class': -1,
                'class_name': f'Carpeta {data_folder} no encontrada',
                'confidence': 0.0,
                'model': 'CNN'
            }]
        
        image_files = [f for f in os.listdir(data_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            return [{
                'image_name': 'N/A',
                'class': -1,
                'class_name': 'No se encontraron imágenes',
                'confidence': 0.0,
                'model': 'CNN'
            }]
        
        print(f"\n🔍 CNN analizando {len(image_files)} imágenes...")
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(data_folder, image_file)
            result = self.predict_image(image_path)
            result['image_name'] = image_file
            results.append(result)
            
            # Mostrar progreso cada 100 imágenes
            if idx % 100 == 0:
                print(f"   Procesadas: {idx}/{len(image_files)}")
        
        print(f"✅ CNN completó el análisis de {len(image_files)} imágenes\n")
        return results
    
    def get_model_info(self):
        """Obtener información del modelo"""
        if not self.is_trained:
            return {
                'status': 'No entrenado',
                'message': 'Ejecuta etiquetador.py para entrenar el modelo'
            }
        
        try:
            metadata_path = "modelos/cnn_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    return json.load(f)
        except:
            pass
        
        return {
            'model_type': 'Convolutional Neural Network (CNN)',
            'img_size': self.img_size,
            'n_classes': len(self.class_names),
            'classes': self.class_names,
            'is_trained': self.is_trained
        }