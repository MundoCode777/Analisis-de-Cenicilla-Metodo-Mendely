# svm_model.py
import os
import json
import joblib
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class SVMModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        # CLASES 1-5 externas (sklearn usa 0-4 interno)
        self.class_names = {
            1: "Clase 1 - Resistente",
            2: "Clase 2 - Moderadamente tolerante",
            3: "Clase 3 - Ligeramente tolerante",
            4: "Clase 4 - Susceptible",
            5: "Clase 5 - Altamente susceptible"
        }
        self.is_trained = False
        self.n_features = 35
        self.load_or_train_model()

    # ============================================================
    # EXTRACCIÓN DE CARACTERÍSTICAS (35)
    # ============================================================
    def extract_features(self, image_path):
        """Extraer 35 características de la imagen para detectar cenicilla."""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                img = img.resize((100, 100))
                img_array = np.array(img, dtype=np.float32)
                features = []

                # Separar canales RGB
                r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

                # 1) COLOR RGB (15)
                for channel in (r, g, b):
                    features.extend([
                        np.mean(channel),
                        np.std(channel),
                        np.median(channel),
                        np.percentile(channel, 25),
                        np.percentile(channel, 75),
                    ])

                # 2) MANCHAS (2)
                whiteness = np.mean((r > 200) & (g > 200) & (b > 200))
                grayness = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b) < 30)
                features.extend([whiteness, grayness])

                # 3) TEXTURA (4)
                gray = 0.299 * r + 0.587 * g + 0.114 * b
                h_diff = np.mean(np.abs(np.diff(gray, axis=1)))
                v_diff = np.mean(np.abs(np.diff(gray, axis=0)))
                local_var = np.var(gray)
                hist_entropy, _ = np.histogram(gray, bins=256, range=(0, 255))
                hist_entropy = hist_entropy / (hist_entropy.sum() + 1e-7)
                entropy = -np.sum(hist_entropy * np.log2(hist_entropy + 1e-7))
                features.extend([h_diff, v_diff, local_var, entropy])

                # 4) CONTRASTE Y BORDES (2)
                contrast = np.std(gray)
                edge_strength = np.mean(np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1]))
                features.extend([contrast, edge_strength])

                # 5) RELACIONES RGB (3)
                rg_ratio = np.mean(r) / (np.mean(g) + 1)
                rb_ratio = np.mean(r) / (np.mean(b) + 1)
                gb_ratio = np.mean(g) / (np.mean(b) + 1)
                features.extend([rg_ratio, rb_ratio, gb_ratio])

                # 6) HISTOGRAMA GRIS 8 BINS (8)
                hist_8, _ = np.histogram(gray, bins=8, range=(0, 255))
                hist_8 = hist_8 / (hist_8.sum() + 1e-7)
                features.extend(hist_8.tolist())

                # 7) UNIFORMIDAD (1)
                color_uniformity = 1 / (np.std(r) + np.std(g) + np.std(b) + 1)
                features.append(color_uniformity)

                result = np.array(features, dtype=np.float32)
                if result.size != self.n_features:
                    print(f"⚠️ Se obtuvieron {result.size} features, se esperaban {self.n_features}")
                    return np.zeros(self.n_features, dtype=np.float32)
                return result

        except Exception as e:
            print(f"⚠️ Error procesando {image_path}: {e}")
            return np.zeros(self.n_features, dtype=np.float32)

    # ============================================================
    # CARGA DE ETIQUETAS
    # ============================================================
    def load_labels(self):
        """Cargar etiquetas desde data/labels.json (externo 1–5)."""
        labels_file = "data/labels.json"
        if os.path.exists(labels_file):
            try:
                with open(labels_file, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                # normalizar a int
                labels = {k: int(v) for k, v in labels.items()}
                print(f"✅ Cargadas {len(labels)} etiquetas desde {labels_file}")
                return labels
            except Exception as e:
                print(f"⚠️ Error cargando etiquetas: {e}")
                return {}
        return {}

    # ============================================================
    # CARGAR O ENTRENAR
    # ============================================================
    def load_or_train_model(self):
        model_path = "modelos/svm_model.pkl"
        scaler_path = "modelos/svm_scaler.pkl"
        models_exist = os.path.exists(model_path) and os.path.exists(scaler_path)
        labels = self.load_labels()
        has_labels = len(labels) >= 10

        if has_labels:
            if models_exist:
                print("\n📊 Modelo existente encontrado")
                print(f"📊 Tienes {len(labels)} imágenes etiquetadas")
                print("\n¿Qué deseas hacer?")
                print("1. Cargar modelo existente")
                print("2. Reentrenar con nuevas etiquetas")
                try:
                    choice = input("Selecciona (1 o 2): ").strip()
                    if choice == "2":
                        self.train_with_real_images(labels)
                        return
                except Exception:
                    pass
            else:
                print(f"\n🎓 Entrenando modelo por primera vez con {len(labels)} imágenes...")
                self.train_with_real_images(labels)
                return

        if models_exist:
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.is_trained = True
                print("✅ Modelo SVM cargado exitosamente")
                return
            except Exception as e:
                print(f"⚠️ Error cargando modelo: {e}")

        print("\n❌ No hay modelo entrenado ni etiquetas disponibles")
        print("📝 Para entrenar el modelo:")
        print("   1. Ejecuta: python etiquetador.py")
        print("   2. Etiqueta al menos 50 imágenes")
        print("   3. Vuelve a ejecutar la aplicación")
        self.is_trained = False

    # ============================================================
    # ENTRENAMIENTO
    # ============================================================
    def train_with_real_images(self, labels=None):
        """Entrenar el SVM con imágenes reales (externo 1–5 → interno 0–4)."""
        if labels is None:
            labels = self.load_labels()

        if not labels or len(labels) < 10:
            print("❌ Necesitas al menos 10 imágenes etiquetadas")
            print("   Ejecuta: python etiquetador.py")
            self.is_trained = False
            return False

        print(f"\n{'='*60}")
        print("📄 ENTRENANDO MODELO SVM CON IMÁGENES REALES")
        print(f"{'='*60}\n")

        X, y, valid_images, failed_images = [], [], [], []
        print(f"📸 Procesando {len(labels)} imágenes...")

        for idx, (image_name, class_id) in enumerate(labels.items(), 1):
            try:
                class_id = int(class_id)  # por si viniera como texto
            except Exception:
                continue

            image_path = os.path.join("data", image_name)
            if not os.path.exists(image_path):
                failed_images.append((image_name, "Archivo no encontrado"))
                continue

            feats = self.extract_features(image_path)
            if feats is not None and not np.all(feats == 0):
                X.append(feats)
                y.append(class_id - 1)  # 1–5 → 0–4
                valid_images.append(image_name)
            else:
                failed_images.append((image_name, "Error extrayendo características"))

            if idx % 50 == 0:
                print(f"   Procesadas: {idx}/{len(labels)}")

        if len(X) < 10:
            print(f"\n❌ ERROR: Solo se procesaron {len(X)} imágenes válidas")
            if failed_images:
                print("⚠️ Imágenes con problemas (primeras 5):")
                for img, reason in failed_images[:5]:
                    print(f"   - {img}: {reason}")
            self.is_trained = False
            return False

        X, y = np.array(X), np.array(y)
        unique, counts = np.unique(y, return_counts=True)
        print("\n📊 Distribución de clases (externo 1–5):")
        for c, cnt in zip(unique, counts):
            print(f"   {self.class_names[c+1]}: {cnt} ({cnt/len(y)*100:.1f}%)")

        # División estratificada si es posible
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except Exception:
            print("⚠️ División sin estratificación (clases muy desbalanceadas)")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Escalado
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Entrenar SVM
        print("\n🧠 Entrenando SVM (RBF, C=10.0, balanced)...")
        self.model = svm.SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(X_train_scaled, y_train)

        # Evaluación
        y_pred_test = self.model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred_test)

        print(f"\n✅ Precisión en prueba: {test_acc:.2%}")
        print("\n📋 Reporte de Clasificación:")
        print(classification_report(
            y_test, y_pred_test,
            target_names=[self.class_names[i+1] for i in sorted(np.unique(y))],
            zero_division=0
        ))
        print("🔍 Matriz de Confusión:")
        print(confusion_matrix(y_test, y_pred_test))

        # Guardar modelo + scaler + metadata
        os.makedirs("modelos", exist_ok=True)
        joblib.dump(self.model, "modelos/svm_model.pkl")
        joblib.dump(self.scaler, "modelos/svm_scaler.pkl")
        metadata = {
            "n_samples": int(len(X)),
            "n_features": int(X.shape[1]),
            "classes": (unique + 1).tolist(),  # externo 1–5
            "test_accuracy": float(test_acc),
            "n_train_samples": int(len(X_train)),
            "n_test_samples": int(len(X_test)),
            "valid_images": valid_images[:100],
        }
        with open("modelos/svm_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("\n💾 Modelo y metadata guardados en carpeta 'modelos/'.")
        self.is_trained = True
        print(f"{'='*60}\n")
        return True

    # ============================================================
    # PREDICCIÓN (devuelve 1–5)
    # ============================================================
    def predict_image(self, image_path):
        """Predecir clase (1–5) de una imagen."""
        if not self.is_trained or self.model is None:
            return {
                "class": -1,
                "class_name": "Modelo no entrenado. Ejecuta etiquetador.py",
                "confidence": 0.0,
                "model": "SVM",
            }
        try:
            feats = self.extract_features(image_path).reshape(1, -1)
            feats_scaled = self.scaler.transform(feats)
            pred_int = self.model.predict(feats_scaled)[0]           # 0–4
            proba = self.model.predict_proba(feats_scaled)[0]
            confidence = float(proba[pred_int])
            return {
                "class": int(pred_int + 1),                           # 1–5
                "class_name": self.class_names[pred_int + 1],
                "confidence": confidence,
                "probabilities": {
                    self.class_names[i + 1]: float(p) for i, p in enumerate(proba)
                },
                "model": "SVM",
            }
        except Exception as e:
            return {
                "class": -1,
                "class_name": f"Error: {e}",
                "confidence": 0.0,
                "model": "SVM",
            }

    # ============================================================
    # ANÁLISIS POR CARPETA
    # ============================================================
    def analyze_dataset(self, data_folder="data"):
        """Analiza todas las imágenes de una carpeta y devuelve lista de predicciones."""
        if not self.is_trained:
            return [{
                "image_name": "N/A",
                "class": -1,
                "class_name": "Modelo no entrenado",
                "confidence": 0.0,
                "model": "SVM",
            }]

        if not os.path.exists(data_folder):
            return [{
                "image_name": "N/A",
                "class": -1,
                "class_name": f"Carpeta {data_folder} no encontrada",
                "confidence": 0.0,
                "model": "SVM",
            }]

        image_files = [f for f in os.listdir(data_folder)
                       if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
        if not image_files:
            return [{
                "image_name": "N/A",
                "class": -1,
                "class_name": "No se encontraron imágenes",
                "confidence": 0.0,
                "model": "SVM",
            }]

        print(f"\n🔍 SVM analizando {len(image_files)} imágenes en '{data_folder}'...")
        results = []
        for idx, image_file in enumerate(image_files, 1):
            image_path = os.path.join(data_folder, image_file)
            result = self.predict_image(image_path)
            result["image_name"] = image_file
            results.append(result)
            if idx % 100 == 0:
                print(f"   Procesadas: {idx}/{len(image_files)}")
        print(f"✅ SVM completó el análisis de {len(image_files)} imágenes\n")
        return results

    # ============================================================
    # INFO DEL MODELO
    # ============================================================
    def get_model_info(self):
        """Información del modelo entrenado."""
        if not self.is_trained:
            return {
                "status": "No entrenado",
                "message": "Ejecuta etiquetador.py para entrenar el modelo",
                "is_trained": False,
            }

        try:
            meta_path = "modelos/svm_metadata.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["is_trained"] = True
                meta["status"] = "Entrenado y listo"
                meta["classes_names"] = self.class_names
                return meta
        except Exception:
            pass

        return {
            "model_type": "Support Vector Machine (SVM)",
            "n_classes": len(self.class_names),
            "classes": self.class_names,
            "is_trained": True,
            "status": "Entrenado (metadata no disponible)",
        }
