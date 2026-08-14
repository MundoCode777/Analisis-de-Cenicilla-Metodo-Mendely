import os
import sys
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modelos.svm_model import SVMModel
from modelos.cnn_model import CNNModel
from modelos.transformer_model import TransformerModel
from modelos.efficientnet_model import EfficientNetModel
from modelos.convnext_model import ConvNeXtModel
from modelos.swin_transformer_model import SwinTransformerModel


def cargar_conteo_etiquetas():
    labels_file = "data/labels.json"
    if not os.path.exists(labels_file):
        return 0
    with open(labels_file, "r", encoding="utf-8") as f:
        return len(json.load(f))


def main():
    print("=" * 60)
    print("🎯 ENTRENADOR DE TODOS LOS MODELOS")
    print("=" * 60)

    n_labels = cargar_conteo_etiquetas()
    print(f"📊 Imágenes etiquetadas disponibles: {n_labels}")
    if n_labels < 10:
        print("❌ Necesitas al menos 10 imágenes etiquetadas (20 para los")
        print("   modelos basados en Transformer). Ejecuta primero:")
        print("   python etiquetador.py")
        return

    modelos = [
        ("SVM",               SVMModel),
        ("CNN",                CNNModel),
        ("Vision Transformer", TransformerModel),
        ("EfficientNet",       EfficientNetModel),
        ("ConvNeXt",           ConvNeXtModel),
        ("Swin Transformer",   SwinTransformerModel),
    ]

    resultados = {}
    for nombre, Clase in modelos:
        print("\n" + "-" * 60)
        print(f"▶ {nombre}")
        print("-" * 60)
        try:
            instancia = Clase()
            resultados[nombre] = "✅ Entrenado/cargado" if instancia.is_trained else "❌ No entrenado"
        except Exception as e:
            resultados[nombre] = f"⚠️ Error: {e}"

    print("\n" + "=" * 60)
    print("📋 RESUMEN")
    print("=" * 60)
    for nombre, estado in resultados.items():
        print(f"   {nombre:20s}: {estado}")
    print("\n🎯 Listo. Ejecuta 'python main.py' para ver los resultados en la app.")


if __name__ == "__main__":
    main()  