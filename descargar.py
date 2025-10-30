import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi

# Config
DATASET_REF = "aryashah2k/mango-leaf-disease-dataset"
DOWNLOAD_DIR = "kaggle_dataset"
TARGET_DIR = "data"
SEARCH_KEY = "Powdery Mildew"  # busca carpetas/archivos que contengan esta palabra (case-insensitive)
IMG_EXT = (".jpg", ".jpeg", ".png")

os.makedirs(TARGET_DIR, exist_ok=True)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Autenticar y descargar + descomprimir el dataset
api = KaggleApi()
try:
    api.authenticate()
except Exception as e:
    raise RuntimeError(f"Kaggle authentication failed: {e}")

print("Descargando y descomprimiendo dataset desde Kaggle...")
try:
    api.dataset_download_files(DATASET_REF, path=DOWNLOAD_DIR, unzip=True)
except Exception as e:
    raise RuntimeError(f"Error descargando dataset: {e}")

# Buscar y copiar imágenes de las rutas que contengan SEARCH_KEY (case-insensitive)
copied = 0
search_key = SEARCH_KEY.lower()
for root, _, files in os.walk(DOWNLOAD_DIR):
    if search_key in root.lower():
        for f in files:
            if f.lower().endswith(IMG_EXT):
                src = os.path.join(root, f)
                dst_name = f
                dst = os.path.join(TARGET_DIR, dst_name)
                # Evitar sobrescribir: si ya existe, añadir sufijo numérico
                if os.path.exists(dst):
                    base, ext = os.path.splitext(dst_name)
                    i = 1
                    while os.path.exists(os.path.join(TARGET_DIR, f"{base}_{i}{ext}")):
                        i += 1
                    dst = os.path.join(TARGET_DIR, f"{base}_{i}{ext}")
                shutil.copy2(src, dst)
                copied += 1

# Si no encontró carpetas con SEARCH_KEY, copiar todas las imágenes como fallback
if copied == 0:
    print(f"No se encontraron rutas que contengan '{SEARCH_KEY}'. Copiando todas las imágenes del dataset...")
    for root, _, files in os.walk(DOWNLOAD_DIR):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                src = os.path.join(root, f)
                dst_name = f
                dst = os.path.join(TARGET_DIR, dst_name)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(dst_name)
                    i = 1
                    while os.path.exists(os.path.join(TARGET_DIR, f"{base}_{i}{ext}")):
                        i += 1
                    dst = os.path.join(TARGET_DIR, f"{base}_{i}{ext}")
                shutil.copy2(src, dst)
                copied += 1

print(f"✅ Descarga y copia completadas. Imágenes guardadas en la carpeta '{TARGET_DIR}' ({copied} archivos).")
