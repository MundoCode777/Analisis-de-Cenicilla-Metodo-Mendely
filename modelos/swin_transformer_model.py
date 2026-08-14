# swin_transformer_model.py
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split


# ============================================================
# CAPAS PERSONALIZADAS DEL SWIN TRANSFORMER
# (mismo estilo que Patches / PatchEncoder en transformer_model.py)
# ============================================================
def window_partition(x, window_size):
    """(B, H, W, C) -> (B*num_windows, window_size, window_size, C)"""
    B = tf.shape(x)[0]
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = tf.reshape(x, [B, H // window_size, window_size, W // window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W):
    """(B*num_windows, window_size, window_size, C) -> (B, H, W, C)"""
    C = windows.shape[-1]
    B = tf.shape(windows)[0] // ((H // window_size) * (W // window_size))
    x = tf.reshape(windows, [B, H // window_size, W // window_size, window_size, window_size, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, C])
    return x


class PatchEmbedding(layers.Layer):
    """Divide la imagen en parches no solapados y los proyecta a embed_dim."""
    def __init__(self, patch_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding="valid")
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        x = self.proj(x)                       # (B, H', W', embed_dim)
        H, W = x.shape[1], x.shape[2]
        C = x.shape[-1]
        B = tf.shape(x)[0]
        x = tf.reshape(x, [B, H * W, C])
        x = self.norm(x)
        return x, H, W

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size, "embed_dim": self.embed_dim})
        return cfg


class SwinTransformerBlock(layers.Layer):
    """
    Bloque Swin: W-MSA (ventanas fijas) o SW-MSA (ventanas desplazadas,
    shift_size > 0) + MLP, con conexiones residuales y pre-normalización.
    """
    def __init__(self, dim, num_heads, window_size=8, shift_size=0, mlp_ratio=2.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads, dropout=0.1)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(int(dim * mlp_ratio), activation=tf.nn.gelu),
            layers.Dropout(0.1),
            layers.Dense(dim),
            layers.Dropout(0.1),
        ])

    def call(self, x, H, W):
        # x: (B, H*W, C)
        B = tf.shape(x)[0]
        C = self.dim
        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, [B, H, W, C])

        if self.shift_size > 0:
            x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])

        windows = window_partition(x, self.window_size)
        windows = tf.reshape(windows, [-1, self.window_size * self.window_size, C])

        attn_windows = self.attn(windows, windows)

        attn_windows = tf.reshape(attn_windows, [-1, self.window_size, self.window_size, C])
        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])

        x = tf.reshape(x, [B, H * W, C])
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim, "num_heads": self.num_heads,
                    "window_size": self.window_size, "shift_size": self.shift_size})
        return cfg


class PatchMerging(layers.Layer):
    """Reduce la resolución a la mitad y duplica el número de canales entre etapas."""
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False)
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, H, W):
        B = tf.shape(x)[0]
        C = x.shape[-1]
        x = tf.reshape(x, [B, H, W, C])
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, [B, (H // 2) * (W // 2), 4 * C])
        x = self.norm(x)
        x = self.reduction(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim})
        return cfg


class SwinClassifier(keras.Model):
    """Swin Transformer ligero (2 etapas) para clasificación de 5 clases de severidad."""
    def __init__(self, patch_size=4, embed_dim=64, depths=(2, 2), num_heads=(2, 4),
                 window_size=8, num_classes=5, **kwargs):
        super().__init__(**kwargs)
        self.rescale = layers.Rescaling(1. / 255)
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)

        self.stage1_blocks = [
            SwinTransformerBlock(embed_dim, num_heads[0], window_size,
                                  shift_size=0 if i % 2 == 0 else window_size // 2)
            for i in range(depths[0])
        ]
        self.merge1 = PatchMerging(embed_dim)

        dim2 = embed_dim * 2
        self.stage2_blocks = [
            SwinTransformerBlock(dim2, num_heads[1], window_size,
                                  shift_size=0 if i % 2 == 0 else window_size // 2)
            for i in range(depths[1])
        ]

        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.pool = layers.GlobalAveragePooling1D()
        self.head = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.rescale(inputs)
        x, H, W = self.patch_embed(x)

        for blk in self.stage1_blocks:
            x = blk(x, H, W)

        x = self.merge1(x, H, W)
        H, W = H // 2, W // 2

        for blk in self.stage2_blocks:
            x = blk(x, H, W)

        x = self.norm(x)
        x = self.pool(x)
        return self.head(x)


# ============================================================
# WRAPPER CON LA MISMA INTERFAZ QUE LOS DEMÁS MODELOS DEL PROYECTO
# ============================================================
class SwinTransformerModel:
    def __init__(self):
        self.img_size = (128, 128)     # 32x32 parches con patch_size=4
        self.patch_size = 4
        self.window_size = 8
        self.model = None
        self.is_trained = False
        self.class_names = {
            1: "Clase 1 - Resistente",
            2: "Clase 2 - Moderadamente tolerante",
            3: "Clase 3 - Ligeramente tolerante",
            4: "Clase 4 - Susceptible",
            5: "Clase 5 - Altamente susceptible"
        }
        self.load_or_create_model()

    def load_labels(self):
        p = "data/labels.json"
        if not os.path.exists(p):
            return {}
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: int(v) for k, v in data.items()}

    def build_architecture(self):
        model = SwinClassifier(
            patch_size=self.patch_size, embed_dim=64, depths=(2, 2),
            num_heads=(2, 4), window_size=self.window_size, num_classes=5,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        # Construir el modelo (necesario para subclassed models antes de
        # guardar/cargar pesos) con un lote de entrada ficticio.
        model(tf.zeros((1, *self.img_size, 3)))
        return model

    def load_images(self, labels):
        X, Y = [], []
        for n, v in labels.items():
            try:
                img = Image.open(os.path.join("data", n)).convert("RGB").resize(self.img_size)
                X.append(np.array(img, dtype=np.float32))
                Y.append(int(v) - 1)
            except Exception:
                pass
        return np.array(X), np.array(Y)

    def create_and_train(self, labels):
        if len(labels) < 20:
            print("❌ Swin Transformer: mínimo 20 imágenes etiquetadas.")
            return False

        X, Y = self.load_images(labels)
        Xtr, Xv, Ytr, Yv = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

        self.model = self.build_architecture()

        cb = [
            callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, verbose=1),
        ]

        print("\n🎓 Entrenando Swin Transformer...")
        self.model.fit(Xtr, Ytr, validation_data=(Xv, Yv), epochs=80,
                        batch_size=16, verbose=1, callbacks=cb)

        val_loss, val_acc = self.model.evaluate(Xv, Yv, verbose=0)
        print(f"✅ Precisión validación: {val_acc:.2%}")

        os.makedirs("modelos", exist_ok=True)
        # Los modelos subclassed de Keras no se guardan de forma fiable en
        # .h5 con model.save(); se guardan solo los PESOS y la arquitectura
        # se reconstruye en load_or_create_model() antes de cargarlos.
        self.model.save_weights("modelos/swin_model.weights.h5")
        with open("modelos/swin_metadata.json", "w") as f:
            json.dump({"val_accuracy": float(val_acc), "samples": len(X)}, f, indent=2)
        self.is_trained = True
        return True

    def load_or_create_model(self):
        weights_path = "modelos/swin_model.weights.h5"
        labels = self.load_labels()
        if os.path.exists(weights_path):
            try:
                self.model = self.build_architecture()
                self.model.load_weights(weights_path)
                self.is_trained = True
                print("✅ Swin Transformer cargado correctamente.")
                return
            except Exception as e:
                print("⚠️ Error cargando pesos:", e)
        if len(labels) >= 20:
            self.create_and_train(labels)
        else:
            print("❌ Swin Transformer: No hay suficientes imágenes etiquetadas (mínimo 20).")

    def preprocess(self, path):
        img = Image.open(path).convert("RGB").resize(self.img_size)
        arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
        return arr

    def predict_image(self, path):
        if not self.is_trained:
            return {"class": -1, "class_name": "Modelo no entrenado", "confidence": 0.0}
        arr = self.preprocess(path)
        pred = self.model.predict(arr, verbose=0)[0]
        c = int(np.argmax(pred))
        conf = float(pred[c])
        return {
            "class": c + 1,
            "class_name": self.class_names[c + 1],
            "confidence": conf,
            "probabilities": {self.class_names[i + 1]: float(p) for i, p in enumerate(pred)},
            "model": "Swin Transformer"
        }

    def analyze_dataset(self, folder="data"):
        imgs = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]
        results = [self.predict_image(os.path.join(folder, f)) | {"image_name": f} for f in imgs]
        print(f"✅ Swin Transformer: Analizadas {len(results)} imágenes.")
        return results