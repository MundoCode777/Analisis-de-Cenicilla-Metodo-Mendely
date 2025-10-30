import os
import requests

# URL de la API de GitHub para listar archivos de esa carpeta
#api_url = "https://api.github.com/repos/DevyanshMalhotra/MangoLeafDiseaseDetection/contents/dataset/Powdery%20Mildew"
api_url = "https://api.github.com/repos/IsaacMwendwa/Omdena-Mango-Leaf-Disease-Detection/contents/src/data/MangoLeafBD_Without_Testset_Augmentation/Test/Powdary%20Mildew"
api_url = "https://api.github.com/repos/IsaacMwendwa/Omdena-Mango-Leaf-Disease-Detection/contents/src/data/MangoLeafBD_Without_Testset_Augmentation/Test/Powdary%20Mildew"

# Crear carpeta "data" si no existe
os.makedirs("data", exist_ok=True)

# Obtener lista de archivos desde GitHub
response = requests.get(api_url)
files = response.json()

for file in files:
    if file["name"].lower().endswith((".jpg", ".jpeg", ".png")):
        print(f"Descargando: {file['name']}...")
        
        # Descargar imagen desde el enlace de descarga directa
        img_data = requests.get(file["download_url"]).content
        
        # Guardar la imagen en la carpeta
        with open(os.path.join("data", file["name"]), "wb") as handler:
            handler.write(img_data)

print("✅ Descarga Completa. Las imágenes están guardadas en la carpeta 'data'")
