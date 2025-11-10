#svm_model.py
import os
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json

class SVMModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = {
            0: "Clase 1 - Resistente",
            1: "Clase 2 - Moderadamente tolerante", 
            2: "Clase 3 - Ligeramente tolerante",
            3: "Clase 4 - Susceptible",
            4: "Clase 5 - Altamente susceptible"
        }
        self.is_trained = False
        self.n_features = 35
        self.load_or_train_model()
    
    def extract_features(self, image_path):
        """Extraer 35 características de la imagen para detectar cenicilla"""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = img.resize((100, 100))
                
                img_array = np.array(img, dtype=np.float32)
                features = []
                
                # Separar canales RGB
                r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
                
                # 1. CARACTERÍSTICAS DE COLOR RGB (15 features)
                # Estadísticas por canal: 5 stats × 3 canales = 15
                for channel in [r, g, b]:
                    features.extend([
                        np.mean(channel),           # Media
                        np.std(channel),            # Desviación estándar
                        np.median(channel),         # Mediana
                        np.percentile(channel, 25), # Cuartil inferior
                        np.percentile(channel, 75), # Cuartil superior
                    ])
                
                # 2. DETECCIÓN DE MANCHAS (2 features)
                # Cenicilla = manchas blancas/grisáceas
                whiteness = np.mean((r > 200) & (g > 200) & (b > 200))
                grayness = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b) < 30)
                features.extend([whiteness, grayness])
                
                # 3. CARACTERÍSTICAS DE TEXTURA (4 features)
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                
                # Diferencias entre píxeles (rugosidad)
                h_diff = np.mean(np.abs(np.diff(gray, axis=1)))
                v_diff = np.mean(np.abs(np.diff(gray, axis=0)))
                
                # Varianza local
                local_var = np.var(gray)
                
                # Entropía (desorden)
                hist_entropy, _ = np.histogram(gray, bins=256, range=(0, 255))
                hist_entropy = hist_entropy / (hist_entropy.sum() + 1e-7)
                entropy = -np.sum(hist_entropy * np.log2(hist_entropy + 1e-7))
                
                features.extend([h_diff, v_diff, local_var, entropy])
                
                # 4. CONTRASTE Y BORDES (2 features)
                contrast = np.std(gray)
                edge_strength = np.mean(np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1]))
                features.extend([contrast, edge_strength])
                
                # 5. RELACIONES ENTRE CANALES RGB (3 features)
                rg_ratio = np.mean(r) / (np.mean(g) + 1)
                rb_ratio = np.mean(r) / (np.mean(b) + 1)
                gb_ratio = np.mean(g) / (np.mean(b) + 1)
                features.extend([rg_ratio, rb_ratio, gb_ratio])
                
                # 6. HISTOGRAMA DE INTENSIDADES (8 features)
                hist_8, _ = np.histogram(gray, bins=8, range=(0, 255))
                hist_8_normalized = hist_8 / (hist_8.sum() + 1e-7)
                features.extend(hist_8_normalized.tolist())
                
                # 7. UNIFORMIDAD DEL COLOR (1 feature)
                color_uniformity = 1 / (np.std(r) + np.std(g) + np.std(b) + 1)
                features.append(color_uniformity)
                
                # Convertir a array y verificar tamaño
                result = np.array(features, dtype=np.float32)
                
                # VERIFICACIÓN CRÍTICA: Debe ser exactamente 35 features
                # 15 (RGB) + 2 (manchas) + 4 (textura) + 2 (contraste) + 3 (ratios) + 8 (hist) + 1 (uniformidad) = 35
                if len(result) != self.n_features:
                    print(f"⚠️ Error: Se obtuvieron {len(result)} features, se esperaban {self.n_features}")
                    return np.zeros(self.n_features, dtype=np.float32)
                
                return result
                
        except Exception as e:
            print(f"⚠️ Error procesando {image_path}: {e}")
            return np.zeros(self.n_features, dtype=np.float32)
    
    def load_labels(self):
        """Cargar etiquetas desde data/labels.json"""
        labels_file = "data/labels.json"
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r') as f:
                    labels = json.load(f)
                    print(f"✅ Cargadas {len(labels)} etiquetas desde {labels_file}")
                    return labels
            except Exception as e:
                print(f"⚠️ Error cargando etiquetas: {e}")
                return {}
        return {}
    
    def load_or_train_model(self):
        """Cargar modelo existente o entrenar con imágenes reales"""
        model_path = "modelos/svm_model.pkl"
        scaler_path = "modelos/svm_scaler.pkl"
        
        # Verificar si existen modelos guardados
        models_exist = os.path.exists(model_path) and os.path.exists(scaler_path)
        
        # Verificar si hay etiquetas disponibles
        labels = self.load_labels()
        has_labels = len(labels) >= 10
        
        # PRIORIDAD: Si hay etiquetas nuevas, ofrecer reentrenamiento
        if has_labels:
            if models_exist:
                print(f"\n📊 Modelo existente encontrado")
                print(f"📊 Tienes {len(labels)} imágenes etiquetadas")
                print(f"\n¿Qué deseas hacer?")
                print(f"1. Cargar modelo existente")
                print(f"2. Reentrenar con nuevas etiquetas")
                
                try:
                    choice = input("Selecciona (1 o 2): ").strip()
                    if choice == "2":
                        self.train_with_real_images(labels)
                        return
                except:
                    pass
            else:
                print(f"\n🎓 Entrenando modelo por primera vez con {len(labels)} imágenes...")
                self.train_with_real_images(labels)
                return
        
        # Intentar cargar modelo existente
        if models_exist:
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                print("✅ Modelo SVM cargado exitosamente")
                return
            except Exception as e:
                print(f"⚠️ Error cargando modelo: {e}")
        
        # Si llegamos aquí, no hay modelo ni etiquetas
        print("\n❌ No hay modelo entrenado ni etiquetas disponibles")
        print("📝 Para entrenar el modelo:")
        print("   1. Ejecuta: python etiquetador.py")
        print("   2. Etiqueta al menos 50 imágenes")
        print("   3. Vuelve a ejecutar la aplicación")
        self.is_trained = False
    
    def train_with_real_images(self, labels=None):
        """Entrenar modelo SVM con imágenes reales etiquetadas"""
        if labels is None:
            labels = self.load_labels()
        
        if not labels or len(labels) < 10:
            print("❌ Necesitas al menos 10 imágenes etiquetadas")
            print("   Ejecuta: python etiquetador.py")
            self.is_trained = False
            return False
        
        print(f"\n{'='*60}")
        print(f"🔄 ENTRENANDO MODELO SVM CON IMÁGENES REALES")
        print(f"{'='*60}\n")
        
        # Extraer características de todas las imágenes etiquetadas
        X = []
        y = []
        valid_images = []
        failed_images = []
        
        print(f"📸 Procesando {len(labels)} imágenes...")
        
        for idx, (image_name, class_id) in enumerate(labels.items(), 1):
            image_path = os.path.join("data", image_name)
            
            if not os.path.exists(image_path):
                failed_images.append((image_name, "Archivo no encontrado"))
                continue
            
            features = self.extract_features(image_path)
            
            if features is not None and not np.all(features == 0):
                X.append(features)
                y.append(class_id)
                valid_images.append(image_name)
            else:
                failed_images.append((image_name, "Error extrayendo características"))
            
            # Mostrar progreso cada 50 imágenes
            if idx % 50 == 0:
                print(f"   Procesadas: {idx}/{len(labels)}")
        
        # Verificar que tenemos suficientes datos
        if len(X) < 10:
            print(f"\n❌ ERROR: Solo se procesaron {len(X)} imágenes válidas")
            print(f"   Se necesitan al menos 10 imágenes para entrenar")
            if failed_images:
                print(f"\n⚠️ Imágenes con problemas:")
                for img, reason in failed_images[:5]:
                    print(f"   - {img}: {reason}")
            self.is_trained = False
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n✅ Datos preparados:")
        print(f"   - Imágenes válidas: {len(X)}")
        print(f"   - Características por imagen: {X.shape[1]}")
        print(f"   - Clases únicas: {np.unique(y)}")
        
        # Mostrar distribución de clases
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n📊 Distribución de clases:")
        for class_id, count in zip(unique, counts):
            percentage = (count / len(y)) * 100
            class_name = self.class_names[class_id]
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        # Verificar si hay suficientes muestras por clase para estratificar
        min_samples = min(counts)
        can_stratify = min_samples >= 2
        
        # Dividir en train/test
        print(f"\n🔀 Dividiendo datos (80% entrenamiento, 20% prueba)...")
        try:
            if can_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                print("   ⚠️ Algunas clases tienen pocas muestras, división sin estratificación")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
        except Exception as e:
            print(f"   ❌ Error en división de datos: {e}")
            self.is_trained = False
            return False
        
        # Escalar características
        print(f"📏 Escalando características...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo SVM
        print(f"\n🧠 Entrenando modelo SVM...")
        print(f"   - Kernel: RBF")
        print(f"   - C: 10.0")
        print(f"   - Balance de clases: Activado")
        
        self.model = svm.SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        print(f"\n📈 Evaluando modelo...")
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print(f"\n{'='*60}")
        print(f"✅ RESULTADOS DEL ENTRENAMIENTO")
        print(f"{'='*60}")
        print(f"📊 Precisión en entrenamiento: {train_acc:.2%}")
        print(f"📊 Precisión en prueba: {test_acc:.2%}")
        
        # Reporte detallado
        print(f"\n📋 Reporte de Clasificación (Conjunto de Prueba):")
        print(f"{'-'*60}")
        try:
            report = classification_report(
                y_test, y_pred_test,
                target_names=[self.class_names[i] for i in sorted(np.unique(y))],
                zero_division=0
            )
            print(report)
        except Exception as e:
            print(f"   ⚠️ No se pudo generar reporte detallado: {e}")
        
        # Matriz de confusión
        print(f"\n🔍 Matriz de Confusión:")
        try:
            cm = confusion_matrix(y_test, y_pred_test)
            print(cm)
        except Exception as e:
            print(f"   ⚠️ No se pudo generar matriz: {e}")
        
        # Guardar modelo y scaler
        print(f"\n💾 Guardando modelo...")
        os.makedirs("modelos", exist_ok=True)
        
        try:
            joblib.dump(self.model, "modelos/svm_model.pkl")
            joblib.dump(self.scaler, "modelos/svm_scaler.pkl")
            
            # Guardar metadata
            metadata = {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'classes': unique.tolist(),
                'train_accuracy': float(train_acc),
                'test_accuracy': float(test_acc),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'valid_images': valid_images[:100]  # Solo las primeras 100
            }
            
            with open("modelos/svm_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Modelo guardado en: modelos/svm_model.pkl")
            print(f"✅ Scaler guardado en: modelos/svm_scaler.pkl")
            print(f"✅ Metadata guardada en: modelos/svm_metadata.json")
            
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")
            self.is_trained = False
            return False
        
        self.is_trained = True
        print(f"\n{'='*60}")
        print(f"🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print(f"{'='*60}\n")
        
        return True
    
    def predict_image(self, image_path):
        """Predecir clase de una imagen"""
        if not self.is_trained or self.model is None:
            return {
                'class': -1,
                'class_name': "Modelo no entrenado. Ejecuta etiquetador.py",
                'confidence': 0.0,
                'model': 'SVM'
            }
        
        try:
            # Extraer características
            features = self.extract_features(image_path)
            features = features.reshape(1, -1)
            
            # Escalar características
            features_scaled = self.scaler.transform(features)
            
            # Predecir
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction]
            
            return {
                'class': int(prediction),
                'class_name': self.class_names[prediction],
                'confidence': float(confidence),
                'probabilities': {
                    self.class_names[i]: float(probabilities[i]) 
                    for i in range(len(probabilities))
                },
                'model': 'SVM'
            }
            
        except Exception as e:
            print(f"❌ Error en predicción SVM: {e}")
            return {
                'class': -1,
                'class_name': f"Error: {str(e)}",
                'confidence': 0.0,
                'model': 'SVM'
            }
    
    def analyze_dataset(self, data_folder="data"):
        """Analizar todas las imágenes en una carpeta"""
        if not self.is_trained:
            return [{
                'image_name': 'N/A',
                'class': -1,
                'class_name': 'Modelo no entrenado',
                'confidence': 0.0,
                'model': 'SVM'
            }]
        
        results = []
        
        if not os.path.exists(data_folder):
            return [{
                'image_name': 'N/A',
                'class': -1,
                'class_name': f'Carpeta {data_folder} no encontrada',
                'confidence': 0.0,
                'model': 'SVM'
            }]
        
        image_files = [f for f in os.listdir(data_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            return [{
                'image_name': 'N/A',
                'class': -1,
                'class_name': 'No se encontraron imágenes',
                'confidence': 0.0,
                'model': 'SVM'
            }]
        
        print(f"\n🔍 SVM analizando {len(image_files)} imágenes...")
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(data_folder, image_file)
            result = self.predict_image(image_path)
            result['image_name'] = image_file
            results.append(result)
            
            # Mostrar progreso cada 100 imágenes
            if idx % 100 == 0:
                print(f"   Procesadas: {idx}/{len(image_files)}")
        
        print(f"✅ SVM completó el análisis de {len(image_files)} imágenes\n")
        return results
    
    def get_model_info(self):
        """Obtener información del modelo"""
        if not self.is_trained:
            return {
                'status': 'No entrenado',
                'message': 'Ejecuta etiquetador.py para entrenar el modelo'
            }
        
        try:
            metadata_path = "modelos/svm_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return metadata
        except:
            pass
        
        return {
            'model_type': 'Support Vector Machine (SVM)',
            'kernel': self.model.kernel if self.model else 'N/A',
            'n_classes': len(self.class_names),
            'classes': self.class_names,
            'is_trained': self.is_trained,
            'n_features': self.n_features
        }