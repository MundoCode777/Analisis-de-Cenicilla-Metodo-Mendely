# transformer_model.py
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    
    def call(self, images):
        batch = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1,1,1,1], 
            padding="VALID"
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch, -1, patch_dims])
    
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, 
            output_dim=projection_dim
        )
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.position_embedding.input_dim, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config

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
        self.is_trained = False
        self.class_names = {
            1: "Clase 1 - Resistente",
            2: "Clase 2 - Moderadamente tolerante",
            3: "Clase 3 - Ligeramente tolerante",
            4: "Clase 4 - Susceptible",
            5: "Clase 5 - Altamente susceptible"
        }
        self.load_or_create_model()

    # ============================================================
    # CARGA DE ETIQUETAS
    # ============================================================
    def load_labels(self):
        """Cargar etiquetas desde data/labels.json (externo 1-5)."""
        labels_file = "data/labels.json"
        if os.path.exists(labels_file):
            try:
                with open(labels_file, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                labels = {k: int(v) for k, v in labels.items()}
                print(f"✅ Transformer: Cargadas {len(labels)} etiquetas desde {labels_file}")
                return labels
            except Exception as e:
                print(f"⚠️ Transformer: Error cargando etiquetas: {e}")
                return {}
        return {}

    # ============================================================
    # CREAR ARQUITECTURA
    # ============================================================
    def create_vit(self):
        """Crear arquitectura Vision Transformer."""
        inputs = layers.Input(shape=(*self.img_size, 3))
        x = layers.Rescaling(1./255)(inputs)
        x = Patches(self.patch_size)(x)
        x = PatchEncoder(self.num_patches, self.projection_dim)(x)
        
        for _ in range(self.transformer_layers):
            x1 = layers.LayerNormalization(epsilon=1e-6)(x)
            att = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.projection_dim,
                dropout=0.1
            )(x1, x1)
            x2 = layers.Add()([att, x])
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            x3 = layers.Dense(self.projection_dim * 2, activation=tf.nn.gelu)(x3)
            x3 = layers.Dropout(0.1)(x3)
            x3 = layers.Dense(self.projection_dim, activation=tf.nn.gelu)(x3)
            x3 = layers.Add()([x3, x2])
            x = x3
        
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        
        for units in self.mlp_head_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(5, activation="softmax")(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # ============================================================
    # CARGA DE IMÁGENES
    # ============================================================
    def load_images_from_labels(self, labels):
        """Cargar imágenes desde etiquetas."""
        X, y, failed = [], [], []
        for idx, (name, label) in enumerate(labels.items(), 1):
            try:
                path = os.path.join("data", name)
                if not os.path.exists(path):
                    failed.append(name)
                    continue
                with Image.open(path) as img:
                    img = img.convert("RGB").resize(self.img_size)
                    X.append(np.array(img, dtype=np.float32))
                    y.append(int(label) - 1)  # 1-5 → 0-4
            except Exception as e:
                failed.append(name)
            if idx % 50 == 0:
                print(f"   Cargadas {idx}/{len(labels)} imágenes...")
        print(f"✅ Transformer: {len(X)} imágenes cargadas, {len(failed)} fallaron.")
        return np.array(X), np.array(y)

    # ============================================================
    # ENTRENAMIENTO
    # ============================================================
    def create_and_train_model(self, labels):
        """Entrenar el modelo Transformer."""
        if not labels or len(labels) < 20:
            print("❌ Transformer: Necesitas al menos 20 imágenes etiquetadas.")
            return False

        X, y = self.load_images_from_labels(labels)
        if len(X) < 20:
            print("❌ Transformer: No hay suficientes imágenes válidas.")
            return False

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("\n🗂️ Construyendo Vision Transformer...")
        self.model = self.create_vit()

        cb = [
            callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=15, 
                restore_best_weights=True, 
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, 
                patience=7, 
                verbose=1
            )
        ]

        print("\n🎓 Entrenando Vision Transformer...")
        self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=80,
            batch_size=16,
            callbacks=cb,
            verbose=1
        )

        print("\n📈 Evaluando modelo...")
        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"✅ Precisión validación: {val_acc:.2%}")

        os.makedirs("modelos", exist_ok=True)
        self.model.save("modelos/transformer_model.h5")
        
        metadata = {
            "samples": len(X),
            "train_split": len(X_train),
            "val_split": len(X_val),
            "val_accuracy": float(val_acc),
            "classes": [1, 2, 3, 4, 5]
        }
        with open("modelos/transformer_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("💾 Modelo y metadata guardados.")
        self.is_trained = True
        return True

    # ============================================================
    # CARGAR O ENTRENAR
    # ============================================================
    def load_or_create_model(self):
        """Cargar modelo existente o entrenar uno nuevo."""
        path = "modelos/transformer_model.h5"
        labels = self.load_labels()
        
        if os.path.exists(path):
            try:
                self.model = keras.models.load_model(
                    path,
                    custom_objects={
                        "Patches": Patches,
                        "PatchEncoder": PatchEncoder
                    }
                )
                self.is_trained = True
                print("✅ Transformer: Modelo cargado exitosamente.")
                return
            except Exception as e:
                print(f"⚠️ Transformer: Error cargando modelo: {e}")
        
        if len(labels) >= 20:
            self.create_and_train_model(labels)
        else:
            print("❌ Transformer: No hay suficientes imágenes etiquetadas (mínimo 20).")

    # ============================================================
    # PREDICCIÓN
    # ============================================================
    def predict_image(self, path):
        """Predecir clase de una imagen."""
        if not self.is_trained or self.model is None:
            return {
                "class": -1,
                "class_name": "Modelo no entrenado",
                "confidence": 0.0,
                "model": "Transformer"
            }
        
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize(self.img_size)
                arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
            
            pred = self.model.predict(arr, verbose=0)[0]
            c = int(np.argmax(pred))
            conf = float(pred[c])
            
            return {
                "class": c + 1,
                "class_name": self.class_names[c + 1],
                "confidence": conf,
                "probabilities": {
                    self.class_names[i+1]: float(p) 
                    for i, p in enumerate(pred)
                },
                "model": "Transformer"
            }
        except Exception as e:
            return {
                "class": -1,
                "class_name": f"Error: {e}",
                "confidence": 0.0,
                "model": "Transformer"
            }

    # ============================================================
    # ANÁLISIS POR CARPETA
    # ============================================================
    def analyze_dataset(self, folder="data"):
        """Analizar todas las imágenes de una carpeta."""
        if not os.path.exists(folder):
            return [{
                "class": -1,
                "class_name": f"Carpeta {folder} no encontrada"
            }]
        
        images = [
            f for f in os.listdir(folder) 
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        
        results = [
            self.predict_image(os.path.join(folder, f)) | {"image_name": f} 
            for f in images
        ]
        
        print(f"✅ Transformer: Analizadas {len(results)} imágenes.")
        return results