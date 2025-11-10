import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import json
import traceback

class Patches(layers.Layer):
    """Capa para dividir imagen en patches (Vision Transformer)"""
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    """Capa para encodear patches con embeddings posicionales"""
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class TransformerModel:
    def __init__(self):
        self.model = None
        self.img_size = (128, 128)
        self.patch_size = 16
        self.num_patches = (self.img_size[0] // self.patch_size) ** 2
        self.projection_dim = 64
        self.num_heads = 4
        self.transformer_layers = 4
        self.mlp_head_units = [128, 64]
        
        self.class_names = {
            0: "Clase 1 - Resistente",
            1: "Clase 2 - Moderadamente tolerante", 
            2: "Clase 3 - Ligeramente tolerante",
            3: "Clase 4 - Susceptible",
            4: "Clase 5 - Altamente susceptible"
        }
        self.is_trained = False
        self.training_history = None
        
        # Verificar GPU
        self.check_gpu()
        self.load_or_create_model()
    
    def check_gpu(self):
        """Verificar disponibilidad de GPU"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Transformer: GPU detectada - {len(gpus)} dispositivo(s)")
            except RuntimeError as e:
                print(f"⚠️ Transformer: Error configurando GPU: {e}")
        else:
            print("⚠️ Transformer: No se detectó GPU, usando CPU")
    
    def load_labels(self):
        """Cargar etiquetas desde data/labels.json"""
        labels_file = "data/labels.json"
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r') as f:
                    labels = json.load(f)
                    print(f"✅ Transformer: Cargadas {len(labels)} etiquetas desde {labels_file}")
                    return labels
            except Exception as e:
                print(f"❌ Transformer: Error cargando etiquetas: {e}")
                return {}
        else:
            print(f"⚠️ Transformer: Archivo {labels_file} no encontrado")
            return {}
    
    def load_or_create_model(self):
        """Cargar modelo existente o crear uno nuevo"""
        model_path = "modelos/transformer_model.h5"
        
        # Verificar si hay etiquetas disponibles
        labels = self.load_labels()
        has_sufficient_labels = len(labels) >= 20  # Mínimo 20 imágenes
        
        # Intentar cargar modelo existente primero
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(
                    model_path,
                    custom_objects={
                        'Patches': Patches, 
                        'PatchEncoder': PatchEncoder
                    },
                    compile=False
                )
                
                # Recompilar el modelo
                self.model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                self.is_trained = True
                print("✅ Transformer: Modelo existente cargado exitosamente")
                
                # Si hay nuevas etiquetas, ofrecer reentrenamiento
                if has_sufficient_labels and len(labels) > 50:
                    print("💡 Transformer: Hay nuevas etiquetas disponibles")
                    print("💡 Ejecuta 'python entrenar_transformer.py' para reentrenar")
                
                return
                
            except Exception as e:
                print(f"⚠️ Transformer: Error cargando modelo existente: {e}")
                print("🔧 Transformer: Creando nuevo modelo...")
        
        # Crear y entrenar nuevo modelo si hay suficientes etiquetas
        if has_sufficient_labels:
            print(f"🎓 Transformer: Iniciando entrenamiento con {len(labels)} imágenes...")
            success = self.create_and_train_model(labels)
            if not success:
                print("❌ Transformer: Falló el entrenamiento inicial")
                self.is_trained = False
        else:
            print(f"❌ Transformer: No hay suficientes imágenes etiquetadas ({len(labels)}/20 mínimas)")
            print("📝 Para entrenar el modelo:")
            print("   1. Ejecuta: python etiquetador.py")
            print("   2. Etiqueta al menos 20 imágenes")
            print("   3. Vuelve a ejecutar la aplicación")
            self.is_trained = False
    
    def mlp(self, x, hidden_units, dropout_rate):
        """Multi-Layer Perceptron"""
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x
    
    def create_vit_model(self):
        """Crear arquitectura Vision Transformer (ViT)"""
        try:
            inputs = layers.Input(shape=(*self.img_size, 3))
            
            # Normalizar imágenes
            normalized = layers.Rescaling(1./255)(inputs)
            
            # Crear patches
            patches = Patches(self.patch_size)(normalized)
            
            # Encodear patches
            encoded_patches = PatchEncoder(self.num_patches, self.projection_dim)(patches)
            
            # Transformer blocks
            for _ in range(self.transformer_layers):
                # Layer normalization 1
                x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
                
                # Multi-head attention
                attention_output = layers.MultiHeadAttention(
                    num_heads=self.num_heads,
                    key_dim=self.projection_dim,
                    dropout=0.1
                )(x1, x1)
                
                # Skip connection 1
                x2 = layers.Add()([attention_output, encoded_patches])
                
                # Layer normalization 2
                x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
                
                # MLP
                x3 = self.mlp(x3, hidden_units=[self.projection_dim * 2, self.projection_dim], dropout_rate=0.1)
                
                # Skip connection 2
                encoded_patches = layers.Add()([x3, x2])
            
            # Layer normalization final
            representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            
            # Global average pooling
            representation = layers.GlobalAveragePooling1D()(representation)
            
            # Dropout
            representation = layers.Dropout(0.3)(representation)
            
            # MLP head
            features = self.mlp(representation, hidden_units=self.mlp_head_units, dropout_rate=0.3)
            
            # Clasificación
            outputs = layers.Dense(5, activation='softmax', name='classifier')(features)
            
            # Crear modelo
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            return model
            
        except Exception as e:
            print(f"❌ Transformer: Error creando modelo: {e}")
            raise
    
    def load_images_from_labels(self, labels):
        """Cargar imágenes desde las etiquetas con manejo robusto de errores"""
        X = []
        y = []
        valid_images = []
        failed_images = []
        
        print(f"📸 Transformer: Cargando {len(labels)} imágenes...")
        
        for idx, (image_name, class_id) in enumerate(labels.items(), 1):
            image_path = os.path.join("data", image_name)
            
            if not os.path.exists(image_path):
                failed_images.append((image_name, "Archivo no encontrado"))
                continue
            
            try:
                # Cargar y preprocesar imagen
                with Image.open(image_path) as img:
                    # Convertir a RGB si es necesario
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Redimensionar
                    img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                    img_array = np.array(img, dtype=np.float32)
                    
                    # Verificar que la imagen sea válida
                    if img_array.size == 0:
                        failed_images.append((image_name, "Imagen vacía"))
                        continue
                    
                    X.append(img_array)
                    y.append(int(class_id))
                    valid_images.append(image_name)
                    
            except Exception as e:
                failed_images.append((image_name, f"Error: {str(e)}"))
            
            # Mostrar progreso
            if idx % 20 == 0:
                print(f"   📸 Cargadas: {idx}/{len(labels)}")
        
        if not X:
            print(f"❌ Transformer: No se pudo cargar ninguna imagen válida")
            if failed_images:
                print("⚠️ Errores encontrados:")
                for img, reason in failed_images[:10]:
                    print(f"   - {img}: {reason}")
            return None, None, valid_images
        
        X_array = np.array(X)
        y_array = np.array(y)
        
        print(f"✅ Transformer: {len(X_array)} imágenes cargadas exitosamente")
        print(f"⚠️ Transformer: {len(failed_images)} imágenes fallaron")
        
        return X_array, y_array, valid_images
    
    def create_and_train_model(self, labels=None):
        """Crear y entrenar modelo Transformer"""
        if labels is None:
            labels = self.load_labels()
        
        if not labels or len(labels) < 20:
            print(f"❌ Transformer: Necesitas al menos 20 imágenes etiquetadas (tienes {len(labels)})")
            self.is_trained = False
            return False
        
        print(f"\n{'='*60}")
        print(f"🔄 TRANSFORMER: INICIANDO ENTRENAMIENTO")
        print(f"{'='*60}")
        print(f"📊 Imágenes disponibles: {len(labels)}")
        print(f"🖼️ Tamaño de imagen: {self.img_size}")
        print(f"🔧 Patch size: {self.patch_size}")
        print(f"{'='*60}\n")
        
        try:
            # Cargar imágenes
            X, y, valid_images = self.load_images_from_labels(labels)
            
            if X is None or len(X) < 20:
                print("❌ Transformer: Insuficientes imágenes válidas para entrenar")
                self.is_trained = False
                return False
            
            # Verificar distribución de clases
            unique, counts = np.unique(y, return_counts=True)
            print(f"\n📊 Distribución de clases:")
            for class_id, count in zip(unique, counts):
                percentage = (count / len(y)) * 100
                class_name = self.class_names[class_id]
                print(f"   {class_name}: {count} imágenes ({percentage:.1f}%)")
            
            # Verificar balance de clases
            min_samples = min(counts)
            if min_samples < 5:
                print(f"⚠️ Transformer: Clase con menos de 5 muestras. Considera agregar más datos.")
            
            # Dividir datos
            print(f"\n🔀 Dividiendo datos...")
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError as e:
                print(f"⚠️ Transformer: No se pudo estratificar, usando división normal: {e}")
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            print(f"   🎯 Entrenamiento: {len(X_train)} imágenes")
            print(f"   📊 Validación: {len(X_val)} imágenes")
            
            # Data Augmentation
            datagen = ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2,
                shear_range=0.1,
                fill_mode='nearest'
            )
            
            # Crear modelo
            print(f"\n🏗️ Construyendo Vision Transformer...")
            self.model = self.create_vit_model()
            
            # Compilar modelo
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    'modelos/transformer_best.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Entrenar modelo
            print(f"\n🎓 Iniciando entrenamiento...")
            print(f"   ⏱️ Épocas: 100 (con early stopping)")
            print(f"   📦 Batch size: 16")
            print(f"   🔄 Data augmentation: Activado")
            
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=16),
                epochs=100,
                validation_data=(X_val, y_val),
                callbacks=callbacks_list,
                verbose=1
            )
            
            self.training_history = history.history
            
            # Evaluar modelo
            print(f"\n📈 Evaluando modelo...")
            train_loss, train_acc = self.model.evaluate(X_train, y_train, verbose=0)
            val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
            
            print(f"✅ Precisión en entrenamiento: {train_acc:.2%}")
            print(f"✅ Precisión en validación: {val_acc:.2%}")
            
            # Guardar modelo final
            print(f"\n💾 Guardando modelo...")
            os.makedirs("modelos", exist_ok=True)
            self.model.save("modelos/transformer_model.h5")
            
            # Guardar metadata
            metadata = {
                'model_type': 'Vision Transformer (ViT)',
                'training_date': str(np.datetime64('now')),
                'n_samples': len(X),
                'n_train': len(X_train),
                'n_val': len(X_val),
                'train_accuracy': float(train_acc),
                'val_accuracy': float(val_acc),
                'train_loss': float(train_loss),
                'val_loss': float(val_loss),
                'epochs_trained': len(history.history['loss']),
                'final_epoch': history.epoch[-1] + 1,
                'img_size': self.img_size,
                'patch_size': self.patch_size,
                'num_patches': self.num_patches,
                'projection_dim': self.projection_dim,
                'num_heads': self.num_heads,
                'transformer_layers': self.transformer_layers,
                'class_distribution': dict(zip(unique.tolist(), counts.tolist())),
                'classes': self.class_names
            }
            
            with open("modelos/transformer_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.is_trained = True
            
            print(f"\n{'='*60}")
            print(f"🎉 ¡TRANSFORMER ENTRENADO EXITOSAMENTE!")
            print(f"📊 Precisión final: {val_acc:.2%}")
            print(f"💾 Modelo guardado: modelos/transformer_model.h5")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Transformer: Error durante el entrenamiento: {e}")
            traceback.print_exc()
            self.is_trained = False
            return False
    
    def preprocess_image(self, image_path):
        """Preprocesar imagen para predicción con manejo robusto de errores"""
        try:
            with Image.open(image_path) as img:
                # Convertir a RGB si es necesario
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Redimensionar
                img = img.resize(self.img_size, Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32)
                
                # Verificar que la imagen no esté vacía
                if img_array.size == 0:
                    raise ValueError("Imagen vacía después del procesamiento")
                
                return np.expand_dims(img_array, axis=0)
                
        except Exception as e:
            print(f"❌ Transformer: Error preprocesando {image_path}: {e}")
            return None
    
    def predict_image(self, image_path):
        """Predecir clase de una imagen individual"""
        if not self.is_trained or self.model is None:
            return {
                'class': -1,
                'class_name': "Modelo Transformer no entrenado",
                'confidence': 0.0,
                'model': 'Transformer',
                'error': 'Modelo no entrenado. Ejecuta el etiquetador primero.'
            }
        
        try:
            processed_img = self.preprocess_image(image_path)
            
            if processed_img is None:
                return {
                    'class': -1,
                    'class_name': "Error al procesar imagen",
                    'confidence': 0.0,
                    'model': 'Transformer',
                    'error': 'No se pudo procesar la imagen'
                }
            
            # Realizar predicción
            predictions = self.model.predict(processed_img, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Crear diccionario de probabilidades
            probabilities = {}
            for i, prob in enumerate(predictions[0]):
                probabilities[self.class_names[i]] = float(prob)
            
            return {
                'class': int(predicted_class),
                'class_name': self.class_names[predicted_class],
                'confidence': confidence,
                'probabilities': probabilities,
                'model': 'Transformer',
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Transformer: Error en predicción: {e}")
            return {
                'class': -1,
                'class_name': f"Error en predicción: {str(e)}",
                'confidence': 0.0,
                'model': 'Transformer',
                'error': str(e)
            }
    
    def analyze_dataset(self, data_folder="data"):
        """Analizar todas las imágenes en una carpeta"""
        if not self.is_trained:
            return [{
                'image_name': 'N/A',
                'class': -1,
                'class_name': 'Modelo Transformer no entrenado',
                'confidence': 0.0,
                'model': 'Transformer',
                'error': 'Ejecuta el etiquetador para entrenar el modelo'
            }]
        
        if not os.path.exists(data_folder):
            return [{
                'image_name': 'N/A',
                'class': -1,
                'class_name': f'Carpeta {data_folder} no encontrada',
                'confidence': 0.0,
                'model': 'Transformer',
                'error': f'Directorio {data_folder} no existe'
            }]
        
        image_files = [f for f in os.listdir(data_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            return [{
                'image_name': 'N/A',
                'class': -1,
                'class_name': 'No se encontraron imágenes',
                'confidence': 0.0,
                'model': 'Transformer',
                'error': 'No hay imágenes en el directorio'
            }]
        
        print(f"\n🔍 Transformer: Analizando {len(image_files)} imágenes en {data_folder}...")
        
        results = []
        successful = 0
        failed = 0
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(data_folder, image_file)
            result = self.predict_image(image_path)
            result['image_name'] = image_file
            results.append(result)
            
            if result.get('success', False):
                successful += 1
            else:
                failed += 1
            
            if idx % 50 == 0:
                print(f"   📊 Progreso: {idx}/{len(image_files)}")
        
        print(f"✅ Transformer: Completado - {successful} éxitos, {failed} fallos")
        
        return results
    
    def get_model_info(self):
        """Obtener información detallada del modelo"""
        if not self.is_trained:
            return {
                'status': 'No entrenado',
                'message': 'Ejecuta el etiquetador para entrenar el modelo',
                'is_trained': False
            }
        
        try:
            metadata_path = "modelos/transformer_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    metadata['is_trained'] = True
                    metadata['status'] = 'Entrenado y listo'
                    return metadata
        except Exception as e:
            print(f"⚠️ Transformer: Error cargando metadata: {e}")
        
        # Información básica si no hay metadata
        return {
            'model_type': 'Vision Transformer (ViT)',
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'num_patches': self.num_patches,
            'n_classes': len(self.class_names),
            'classes': self.class_names,
            'is_trained': True,
            'status': 'Entrenado (metadata no disponible)'
        }
    
    def get_training_history(self):
        """Obtener historial de entrenamiento"""
        return self.training_history