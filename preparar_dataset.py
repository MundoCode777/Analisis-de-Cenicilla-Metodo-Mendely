"""
Convierte la estructura data/Clase-1, data/Clase-2, ..., data/Clase-5
en el formato que usan tus modelos: una carpeta data/ plana +
data/labels.json (el mismo formato que genera etiquetador.py).

Copia las imágenes (no las mueve, para no perder el orden original)
hacia data/, renombrándolas con un prefijo de clase para evitar
que se sobrescriban archivos con el mismo nombre entre carpetas.
"""

import os
import json
import shutil

CARPETA_DATA = "data"
EXTENSIONES = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

def main():
    labels = {}
    total_copiadas = 0

    for clase_num in range(1, 6):
        carpeta_clase = os.path.join(CARPETA_DATA, f"Clase-{clase_num}")

        if not os.path.exists(carpeta_clase):
            print(f"⚠️ No existe {carpeta_clase}, se omite.")
            continue

        imagenes = [f for f in os.listdir(carpeta_clase)
                    if f.lower().endswith(EXTENSIONES)]

        print(f"📁 Clase-{clase_num}: {len(imagenes)} imágenes encontradas")

        for img_name in imagenes:
            origen = os.path.join(carpeta_clase, img_name)

            # Prefijo para evitar colisiones de nombres entre clases
            nuevo_nombre = f"clase{clase_num}_{img_name}"
            destino = os.path.join(CARPETA_DATA, nuevo_nombre)

            shutil.copy2(origen, destino)
            labels[nuevo_nombre] = clase_num
            total_copiadas += 1

    labels_file = os.path.join(CARPETA_DATA, "labels.json")
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)

    print(f"\n✅ Listo. {total_copiadas} imágenes copiadas a '{CARPETA_DATA}/'")
    print(f"💾 Etiquetas guardadas en: {labels_file}")
    print(f"\n📊 Distribución final:")
    for clase_num in range(1, 6):
        count = sum(1 for v in labels.values() if v == clase_num)
        print(f"   Clase {clase_num}: {count} imágenes")

    print(f"\n🎯 Ahora puedes ejecutar: python entrenar_todos.py")

if __name__ == "__main__":
    main()