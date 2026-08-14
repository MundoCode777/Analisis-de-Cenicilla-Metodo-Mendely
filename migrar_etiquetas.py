import json
import os

def migrate_labels():
    labels_file = "data/labels.json"
    
    if not os.path.exists(labels_file):
        print("❌ No se encontró el archivo data/labels.json")
        return
    
    # Cargar etiquetas
    with open(labels_file, 'r') as f:
        labels = json.load(f)
    
    print(f"📊 Etiquetas encontradas: {len(labels)}")
    
    # Verificar si necesita migración
    needs_migration = False
    for img_name, class_id in labels.items():
        if class_id in [0, 1, 2, 3, 4]:
            needs_migration = True
            break
    
    if not needs_migration:
        print("✅ Las etiquetas ya están en formato 1-5")
        return
    
    # Hacer backup
    backup_file = "data/labels_backup.json"
    with open(backup_file, 'w') as f:
        json.dump(labels, f, indent=2)
    print(f"💾 Backup creado: {backup_file}")
    
    # Migrar etiquetas
    migrated = {}
    for img_name, class_id in labels.items():
        if class_id in [0, 1, 2, 3, 4]:
            migrated[img_name] = class_id + 1
        else:
            migrated[img_name] = class_id
    
    # Guardar etiquetas migradas
    with open(labels_file, 'w') as f:
        json.dump(migrated, f, indent=2)
    
    print("✅ Etiquetas migradas exitosamente de 0-4 a 1-5")
    print(f"📁 Guardado en: {labels_file}")
    
    # Mostrar distribución
    stats = {}
    for class_id in migrated.values():
        stats[class_id] = stats.get(class_id, 0) + 1
    
    print("\n📊 Nueva distribución:")
    for class_id in sorted(stats.keys()):
        print(f"   Clase {class_id}: {stats[class_id]} imágenes")

if __name__ == "__main__":
    migrate_labels()