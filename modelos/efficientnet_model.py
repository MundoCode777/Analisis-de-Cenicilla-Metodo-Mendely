# efficientnet_model.py
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


class EfficientNetModel:
    """
    Clasificador de severidad de cenicilla basado en EfficientNetB0,
    entrenado DESDE CERO (weights=None), siguiendo el mismo patrón
    de entrenamiento/serialización que cnn_model.py.
    """

    def __init__(self):
        self.model = None
        self.img_size = (224, 224)  # tamaño nativo de EfficientNetB0
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
        labels_file = "data/labels.json"
        if os.path.exists(labels_file):
            try:
                with open(labels_file, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                labels = {k: int(v) for k, v in labels.items()}
                print(f"✅ EfficientNet: Cargadas {len(labels)} etiquetas desde {labels_file}")
                return labels
            except Exception as e:
                print(f"⚠️ EfficientNet: Error cargando etiquetas: {e}")
                return {}
        return {}

    # ============================================================
    # ARQUITECTURA (sin pesos preentrenados)
    # ============================================================
    def create_model_architecture(self):
        base = EfficientNetB0(
            include_top=True,
            weights=None,              # <- entrenamiento desde cero
            classes=5,
            classifier_activation="softmax",
            input_shape=(*self.img_size, 3),
        )
        model = keras.Sequential([base])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ============================================================
    # CARGA DE IMÁGENES
    # ============================================================
    def load_images_from_labels(self, labels):
        X, y, failed = [], [], []
        for idx, (name, label) in enumerate(labels.items(), 1):
            try:
                path = os.path.join("data", name)
                if not os.path.exists(path):
                    failed.append(name)
                    continue
                with Image.open(path) as img:
                    img = img.convert("RGB").resize(self.img_size)
                    X.append(np.array(img, dtype=np.float32) / 255.0)
                    y.append(int(label) - 1)  # 1-5 -> 0-4
            except Exception:
                failed.append(name)
            if idx % 50 == 0:
                print(f"   Cargadas {idx}/{len(labels)} imágenes...")
        print(f"✅ EfficientNet: {len(X)} imágenes cargadas, {len(failed)} fallaron.")
        return np.array(X), np.array(y)

    # ============================================================
    # ENTRENAMIENTO
    # ============================================================
    def create_and_train_model(self, labels):
        if not labels or len(labels) < 10:
            print("❌ EfficientNet: Necesitas al menos 10 imágenes etiquetadas.")
            return False

        X, y = self.load_images_from_labels(labels)
        if len(X) < 10:
            print("❌ EfficientNet: No hay suficientes imágenes válidas.")
            return False

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        datagen = ImageDataGenerator(
            rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
            zoom_range=0.2, horizontal_flip=True
        )

        print("\n🏗️ Construyendo EfficientNetB0 (desde cero)...")
        self.model = self.create_model_architecture()

        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1),
        ]

        print("\n🎓 Entrenando EfficientNetB0...")
        self.model.fit(
            datagen.flow(X_train, y_train, batch_size=16),
            validation_data=(X_val, y_val),
            epochs=30, callbacks=cb, verbose=1,
        )

        val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"✅ Precisión validación: {val_acc:.2%}")

        os.makedirs("modelos", exist_ok=True)
        self.model.save("modelos/efficientnet_model.h5")
        metadata = {
            "samples": len(X), "train_split": len(X_train), "val_split": len(X_val),
            "val_accuracy": float(val_acc), "classes": [1, 2, 3, 4, 5],
        }
        with open("modelos/efficientnet_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print("💾 Modelo y metadata guardados.")
        self.is_trained = True
        return True

    # ============================================================
    # CARGAR O ENTRENAR
    # ============================================================
    def load_or_create_model(self):
        path = "modelos/efficientnet_model.h5"
        labels = self.load_labels()
        if os.path.exists(path):
            try:
                self.model = keras.models.load_model(path)
                self.is_trained = True
                print("✅ EfficientNet: Modelo cargado exitosamente.")
                return
            except Exception as e:
                print(f"⚠️ EfficientNet: Error cargando modelo: {e}")
        if len(labels) >= 10:
            self.create_and_train_model(labels)
        else:
            print("❌ EfficientNet: No hay suficientes imágenes etiquetadas (mínimo 10).")

    # ============================================================
    # PREDICCIÓN
    # ============================================================
    def predict_image(self, path):
        if not self.is_trained:
            return {"class": -1, "class_name": "Modelo no entrenado", "confidence": 0.0}
        try:
            with Image.open(path) as img:
                img = img.convert("RGB").resize(self.img_size)
                arr = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, 0)
            pred = self.model.predict(arr, verbose=0)[0]
            c = int(np.argmax(pred))
            conf = float(pred[c])
            return {
                "class": c + 1,
                "class_name": self.class_names[c + 1],
                "confidence": conf,
                "probabilities": {self.class_names[i + 1]: float(p) for i, p in enumerate(pred)},
                "model": "EfficientNet",
            }
        except Exception as e:
            return {"class": -1, "class_name": f"Error: {e}", "confidence": 0.0, "model": "EfficientNet"}

    # ============================================================
    # ANÁLISIS POR CARPETA
    # ============================================================
    def analyze_dataset(self, folder="data"):
        if not os.path.exists(folder):
            return [{"class": -1, "class_name": f"Carpeta {folder} no encontrada"}]
        images = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
        results = [self.predict_image(os.path.join(folder, f)) | {"image_name": f} for f in images]
        print(f"✅ EfficientNet: Analizadas {len(results)} imágenes.")
        return results