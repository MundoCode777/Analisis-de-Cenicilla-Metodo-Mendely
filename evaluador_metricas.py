# evaluador_metricas.py
"""
Módulo para calcular métricas de evaluación de modelos
Incluye: Exactitud, Precisión, Recall (Exhaustividad), F1-Score
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import json
import os

class MetricsEvaluator:
    def __init__(self):
        self.class_names = {
            0: "Clase 0 - Resistente",
            1: "Clase 1 - Moderadamente tolerante", 
            2: "Clase 2 - Ligeramente tolerante",
            3: "Clase 3 - Susceptible",
            4: "Clase 4 - Altamente susceptible"
        }
    
    def load_labels(self):
        """Cargar etiquetas verdaderas desde labels.json"""
        labels_file = "data/labels.json"
        if os.path.exists(labels_file):
            try:
                with open(labels_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def calculate_metrics(self, predictions, model_name="Modelo"):
        """
        Calcular métricas de evaluación comparando predicciones con etiquetas reales
        
        Args:
            predictions: Lista de diccionarios con predicciones del modelo
                        Cada dict debe tener: {'image_name': str, 'class': int, ...}
            model_name: Nombre del modelo para el reporte
        
        Returns:
            dict con todas las métricas calculadas
        """
        # Cargar etiquetas verdaderas
        true_labels = self.load_labels()
        
        if not true_labels:
            return {
                'error': 'No se encontraron etiquetas verdaderas (labels.json)',
                'message': 'Ejecuta etiquetador.py primero'
            }
        
        # Preparar datos para comparación
        y_true = []
        y_pred = []
        matched_images = []
        
        for pred in predictions:
            image_name = pred.get('image_name', '')
            
            # Verificar si esta imagen tiene etiqueta real
            if image_name in true_labels:
                true_class = true_labels[image_name]
                pred_class = pred.get('class', -1)
                
                if pred_class != -1:  # Predicción válida
                    y_true.append(true_class)
                    y_pred.append(pred_class)
                    matched_images.append(image_name)
        
        # Verificar que tengamos datos suficientes
        if len(y_true) < 5:
            return {
                'error': 'Datos insuficientes',
                'message': f'Solo {len(y_true)} imágenes tienen etiquetas y predicciones válidas',
                'n_samples': len(y_true)
            }
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calcular métricas principales
        try:
            # 1. EXACTITUD (Accuracy)
            accuracy = accuracy_score(y_true, y_pred)
            
            # 2. PRECISIÓN (Precision) - Promedio ponderado
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            
            # 3. EXHAUSTIVIDAD (Recall/Sensitivity)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            
            # 4. F1-SCORE
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            # 5. MATRIZ DE CONFUSIÓN
            cm = confusion_matrix(y_true, y_pred)
            
            # Construir reporte detallado
            metrics = {
                'model_name': model_name,
                'n_samples': len(y_true),
                'n_matched': len(matched_images),
                
                # Métricas globales
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'precision_weighted': float(precision_weighted),
                'recall_macro': float(recall_macro),
                'recall_weighted': float(recall_weighted),
                'f1_macro': float(f1_macro),
                'f1_weighted': float(f1_weighted),
                
                # Métricas por clase
                'per_class': {},
                
                # Matriz de confusión
                'confusion_matrix': cm.tolist(),
                
                # Imágenes analizadas
                'matched_images': matched_images[:10]  # Solo las primeras 10
            }
            
            # Métricas por clase
            unique_classes = np.unique(np.concatenate([y_true, y_pred]))
            for class_id in unique_classes:
                if class_id < len(precision_per_class):
                    metrics['per_class'][int(class_id)] = {
                        'class_name': self.class_names.get(class_id, f'Clase {class_id}'),
                        'precision': float(precision_per_class[class_id]),
                        'recall': float(recall_per_class[class_id]),
                        'f1_score': float(f1_per_class[class_id]),
                        'support': int(np.sum(y_true == class_id))
                    }
            
            return metrics
            
        except Exception as e:
            return {
                'error': f'Error calculando métricas: {str(e)}',
                'n_samples': len(y_true)
            }
    
    def print_metrics_report(self, metrics):
        """Imprimir reporte formateado de métricas"""
        if 'error' in metrics:
            print(f"\n❌ {metrics['error']}")
            if 'message' in metrics:
                print(f"   {metrics['message']}")
            return
        
        print(f"\n{'='*70}")
        print(f"📊 REPORTE DE MÉTRICAS - {metrics['model_name']}")
        print(f"{'='*70}")
        
        print(f"\n📈 Muestras analizadas: {metrics['n_samples']}")
        
        print(f"\n🎯 MÉTRICAS GLOBALES:")
        print(f"{'─'*70}")
        print(f"  Exactitud (Accuracy):              {metrics['accuracy']:.2%}")
        print(f"  Precisión (Macro):                 {metrics['precision_macro']:.2%}")
        print(f"  Precisión (Weighted):              {metrics['precision_weighted']:.2%}")
        print(f"  Exhaustividad/Recall (Macro):      {metrics['recall_macro']:.2%}")
        print(f"  Exhaustividad/Recall (Weighted):   {metrics['recall_weighted']:.2%}")
        print(f"  F1-Score (Macro):                  {metrics['f1_macro']:.2%}")
        print(f"  F1-Score (Weighted):               {metrics['f1_weighted']:.2%}")
        
        print(f"\n📋 MÉTRICAS POR CLASE:")
        print(f"{'─'*70}")
        
        for class_id, class_metrics in sorted(metrics['per_class'].items()):
            print(f"\n  {class_metrics['class_name']}:")
            print(f"    • Precisión:      {class_metrics['precision']:.2%}")
            print(f"    • Recall:         {class_metrics['recall']:.2%}")
            print(f"    • F1-Score:       {class_metrics['f1_score']:.2%}")
            print(f"    • Soporte:        {class_metrics['support']} muestras")
        
        print(f"\n🔍 MATRIZ DE CONFUSIÓN:")
        print(f"{'─'*70}")
        cm = np.array(metrics['confusion_matrix'])
        print("     ", end="")
        for i in range(len(cm)):
            print(f"C{i}  ", end="")
        print()
        for i, row in enumerate(cm):
            print(f"  C{i} ", end="")
            for val in row:
                print(f"{val:3d} ", end="")
            print()
        
        print(f"\n{'='*70}\n")
    
    def compare_models(self, all_metrics):
        """
        Comparar métricas entre múltiples modelos
        
        Args:
            all_metrics: dict con formato {'SVM': metrics_dict, 'CNN': metrics_dict, ...}
        """
        print(f"\n{'='*70}")
        print(f"🏆 COMPARATIVA ENTRE MODELOS")
        print(f"{'='*70}\n")
        
        # Tabla comparativa
        print(f"{'Métrica':<30} {'SVM':>12} {'CNN':>12} {'Transformer':>12}")
        print(f"{'─'*70}")
        
        metrics_to_compare = [
            ('Exactitud', 'accuracy'),
            ('Precisión (Weighted)', 'precision_weighted'),
            ('Recall (Weighted)', 'recall_weighted'),
            ('F1-Score (Weighted)', 'f1_weighted')
        ]
        
        for label, key in metrics_to_compare:
            print(f"{label:<30}", end="")
            
            values = []
            for model_name in ['SVM', 'CNN', 'Transformer']:
                if model_name in all_metrics and key in all_metrics[model_name]:
                    value = all_metrics[model_name][key]
                    values.append(value)
                    print(f"{value:>11.2%} ", end="")
                else:
                    print(f"{'N/A':>12} ", end="")
            print()
            
            # Marcar el mejor
            if values:
                best_idx = np.argmax(values)
                model_names = ['SVM', 'CNN', 'Transformer']
                if best_idx < len(model_names):
                    best_model = model_names[best_idx]
                    print(f"{'':>30} ⭐ Mejor: {best_model}")
        
        print(f"\n{'='*70}\n")
    
    def save_metrics_to_file(self, metrics, filename="modelos/metrics_report.json"):
        """Guardar métricas en archivo JSON"""
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"✅ Métricas guardadas en: {filename}")
        except Exception as e:
            print(f"⚠️ Error guardando métricas: {e}")


# Función auxiliar para uso fácil
def evaluate_model_predictions(predictions, model_name="Modelo"):
    """
    Función de conveniencia para evaluar predicciones
    
    Ejemplo de uso:
        from evaluador_metricas import evaluate_model_predictions
        
        results = model.analyze_dataset("data")
        metrics = evaluate_model_predictions(results, "SVM")
    """
    evaluator = MetricsEvaluator()
    metrics = evaluator.calculate_metrics(predictions, model_name)
    evaluator.print_metrics_report(metrics)
    return metrics