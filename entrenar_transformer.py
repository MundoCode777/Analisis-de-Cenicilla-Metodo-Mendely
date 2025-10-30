"""
Script para entrenar manualmente el modelo Transformer
"""

import os
import sys

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modelos.transformer_model import TransformerModel

def main():
    print("🎯 ENTRENADOR MANUAL DE TRANSFORMER")
    print("=" * 50)
    
    # Crear instancia del modelo
    transformer = TransformerModel()
    
    # Cargar etiquetas
    labels = transformer.load_labels()
    
    if not labels or len(labels) < 20:
        print(f"❌ No hay suficientes imágenes etiquetadas ({len(labels)}/20 mínimas)")
        print("💡 Ejecuta primero: python etiquetador.py")
        return
    
    print(f"📊 Imágenes disponibles: {len(labels)}")
    
    # Preguntar si quiere entrenar
    respuesta = input("¿Deseas entrenar el modelo Transformer? (s/n): ").strip().lower()
    
    if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
        print("\n🚀 Iniciando entrenamiento...")
        success = transformer.create_and_train_model(labels)
        
        if success:
            print("✅ Entrenamiento completado exitosamente!")
        else:
            print("❌ El entrenamiento falló. Revisa los mensajes de error.")
    else:
        print("❌ Entrenamiento cancelado")

if __name__ == "__main__":
    main()