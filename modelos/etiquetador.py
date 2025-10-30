# etiquetador.py
"""
Herramienta de Etiquetado de Imágenes para Análisis de Cenicilla
Permite etiquetar imágenes en 5 clases de severidad
"""

import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import os
import json
import platform

DEFAULT_FONT = "Segoe UI" if platform.system() == "Windows" else "Helvetica"

class ImageLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Etiquetador de Imágenes - Cenicilla")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        self.root.config(bg="#F8FAFF")
        
        self.data_folder = "data"
        self.labels_file = "data/labels.json"
        
        # Verificar que existe la carpeta data
        if not os.path.exists(self.data_folder):
            messagebox.showerror(
                "Error", 
                "La carpeta 'data' no existe.\n\n"
                "Crea una carpeta llamada 'data' y coloca allí las imágenes de hojas."
            )
            root.destroy()
            return
        
        # Cargar imágenes
        self.images = [f for f in os.listdir(self.data_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not self.images:
            messagebox.showerror(
                "Error", 
                "No hay imágenes en la carpeta 'data'.\n\n"
                "Coloca las imágenes de hojas en la carpeta 'data'."
            )
            root.destroy()
            return
        
        print(f"✅ Encontradas {len(self.images)} imágenes en la carpeta 'data'")
        
        # Cargar etiquetas existentes
        self.labels = self.load_labels()
        self.current_index = 0
        
        # Información de las clases
        self.class_info = {
            0: ("Clase 0 - Resistente", "#10B981", "Sin síntomas o muy leves (<5%)"),
            1: ("Clase 1 - Moderadamente Tolerante", "#84CC16", "Síntomas leves (5-25%)"),
            2: ("Clase 2 - Ligeramente Tolerante", "#F59E0B", "Síntomas moderados (25-50%)"),
            3: ("Clase 3 - Susceptible", "#F97316", "Síntomas severos (50-75%)"),
            4: ("Clase 4 - Altamente Susceptible", "#EF4444", "Síntomas muy severos (>75%)")
        }
        
        self.create_widgets()
        self.load_image()
        
        # Atajos de teclado
        self.root.bind('0', lambda e: self.assign_class(0))
        self.root.bind('1', lambda e: self.assign_class(1))
        self.root.bind('2', lambda e: self.assign_class(2))
        self.root.bind('3', lambda e: self.assign_class(3))
        self.root.bind('4', lambda e: self.assign_class(4))
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<space>', lambda e: self.next_image())
    
    def load_labels(self):
        """Cargar etiquetas existentes desde JSON"""
        if os.path.exists(self.labels_file):
            try:
                with open(self.labels_file, 'r') as f:
                    labels = json.load(f)
                    print(f"✅ Cargadas {len(labels)} etiquetas existentes")
                    return labels
            except Exception as e:
                print(f"⚠️ Error cargando etiquetas: {e}")
                return {}
        return {}
    
    def save_labels(self):
        """Guardar etiquetas en archivo JSON"""
        try:
            os.makedirs(self.data_folder, exist_ok=True)
            with open(self.labels_file, 'w') as f:
                json.dump(self.labels, f, indent=2)
        except Exception as e:
            print(f"❌ Error guardando etiquetas: {e}")
            messagebox.showerror("Error", f"No se pudieron guardar las etiquetas:\n{e}")
    
    def create_widgets(self):
        # ========== HEADER ==========
        header = tk.Frame(self.root, bg="#6366F1", height=80)
        header.pack(fill="x")
        
        title = tk.Label(
            header, 
            text="🌿 Etiquetador de Imágenes - Análisis de Cenicilla", 
            font=(DEFAULT_FONT, 20, "bold"),
            bg="#6366F1", 
            fg="white"
        )
        title.pack(pady=25)
        
        # ========== MAIN CONTAINER ==========
        main = tk.Frame(self.root, bg="#F8FAFF")
        main.pack(fill="both", expand=True, padx=20, pady=15)
        
        # ========== LEFT PANEL - IMAGEN ==========
        left_panel = tk.Frame(
            main, 
            bg="white", 
            highlightbackground="#E0E7FF", 
            highlightthickness=2
        )
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Barra de progreso
        progress_frame = tk.Frame(left_panel, bg="white")
        progress_frame.pack(fill="x", pady=10)
        
        self.progress_label = tk.Label(
            progress_frame, 
            text="", 
            font=(DEFAULT_FONT, 12, "bold"),
            bg="white", 
            fg="#475569"
        )
        self.progress_label.pack()
        
        # Canvas para la imagen con scroll
        canvas_container = tk.Frame(left_panel, bg="white")
        canvas_container.pack(expand=True, fill="both", padx=15, pady=10)
        
        # Crear canvas con scrollbars
        self.canvas = tk.Canvas(canvas_container, bg="white", highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout para el canvas y scrollbars
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)
        
        # Frame interior para la imagen
        self.image_frame = tk.Frame(self.canvas, bg="white")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.image_frame, anchor="nw")
        
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack(expand=True, fill="both")
        
        # Configurar el scroll
        self.image_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Nombre de archivo
        self.name_label = tk.Label(
            left_panel, 
            text="", 
            font=(DEFAULT_FONT, 11),
            bg="white", 
            fg="#64748b",
            wraplength=600
        )
        self.name_label.pack(pady=10)
        
        # Botones de navegación
        nav_frame = tk.Frame(left_panel, bg="white")
        nav_frame.pack(fill="x", padx=15, pady=15)
        
        prev_btn = tk.Button(
            nav_frame, 
            text="◀ Anterior (←)",
            font=(DEFAULT_FONT, 11),
            bg="#E0E7FF", 
            fg="#4F46E5",
            activebackground="#C7D2FE",
            relief="flat", 
            pady=12,
            cursor="hand2",
            command=self.previous_image
        )
        prev_btn.pack(side="left", fill="x", expand=True, padx=(0, 5))
        
        next_btn = tk.Button(
            nav_frame, 
            text="Siguiente (→)",
            font=(DEFAULT_FONT, 11),
            bg="#E0E7FF", 
            fg="#4F46E5",
            activebackground="#C7D2FE",
            relief="flat", 
            pady=12,
            cursor="hand2",
            command=self.next_image
        )
        next_btn.pack(side="right", fill="x", expand=True, padx=(5, 0))
        
        # ========== RIGHT PANEL - CLASIFICACIÓN ==========
        right_panel = tk.Frame(
            main, 
            bg="white", 
            highlightbackground="#E0E7FF",
            highlightthickness=2, 
            width=380
        )
        right_panel.pack(side="right", fill="y")
        right_panel.pack_propagate(False)
        
        # Crear un frame con scroll para los botones de clasificación
        right_scroll_frame = tk.Frame(right_panel, bg="white")
        right_scroll_frame.pack(fill="both", expand=True)
        
        # Canvas para el panel derecho
        right_canvas = tk.Canvas(right_scroll_frame, bg="white", highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_scroll_frame, orient="vertical", command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scrollbar.set)
        
        right_scrollbar.pack(side="right", fill="y")
        right_canvas.pack(side="left", fill="both", expand=True)
        
        # Frame interior para el contenido
        right_content = tk.Frame(right_canvas, bg="white")
        right_canvas.create_window((0, 0), window=right_content, anchor="nw")
        
        # Instrucciones
        instructions_frame = tk.Frame(right_content, bg="#DDD6FE")
        instructions_frame.pack(fill="x", pady=(0, 15))
        
        instructions = tk.Label(
            instructions_frame, 
            text="Selecciona la clase de severidad:",
            font=(DEFAULT_FONT, 13, "bold"),
            bg="#DDD6FE", 
            fg="#5B21B6"
        )
        instructions.pack(pady=12)
        
        # Botones de clase
        self.class_buttons = []
        for class_id in range(5):
            name, color, desc = self.class_info[class_id]
            
            btn_frame = tk.Frame(right_content, bg="white")
            btn_frame.pack(fill="x", padx=12, pady=6)
            
            btn = tk.Button(
                btn_frame, 
                text=f"{name}\n{desc}",
                font=(DEFAULT_FONT, 10),
                bg=color, 
                fg="white",
                activebackground=self.adjust_color_brightness(color, 0.9),
                activeforeground="white",
                relief="flat",
                pady=15,
                cursor="hand2",
                wraplength=340,
                justify="left",
                command=lambda c=class_id: self.assign_class(c)
            )
            btn.pack(fill="x")
            
            # Agregar atajo de teclado visible
            shortcut = tk.Label(
                btn_frame,
                text=f"Atajo: {class_id}",
                font=(DEFAULT_FONT, 8),
                bg="white",
                fg="#94a3b8"
            )
            shortcut.pack(pady=2)
            
            self.class_buttons.append(btn)
        
        # Estado actual de etiquetado
        self.current_label_frame = tk.Frame(right_content, bg="#FEF3C7")
        self.current_label_frame.pack(fill="x", padx=12, pady=12)
        
        current_title = tk.Label(
            self.current_label_frame,
            text="Estado Actual:",
            font=(DEFAULT_FONT, 10, "bold"),
            bg="#FEF3C7",
            fg="#92400E"
        )
        current_title.pack(pady=(8, 2))
        
        self.current_label_text = tk.Label(
            self.current_label_frame,
            text="⚠ Sin etiquetar",
            font=(DEFAULT_FONT, 11, "bold"),
            bg="#FEF3C7", 
            fg="#92400E"
        )
        self.current_label_text.pack(pady=(0, 8))
        
        # Estadísticas
        stats_frame = tk.Frame(right_content, bg="white")
        stats_frame.pack(fill="x", padx=12, pady=10)
        
        stats_title = tk.Label(
            stats_frame,
            text="📊 Estadísticas",
            font=(DEFAULT_FONT, 11, "bold"),
            bg="white",
            fg="#0f172a"
        )
        stats_title.pack(pady=(0, 5))
        
        self.stats_label = tk.Label(
            stats_frame, 
            text="",
            font=(DEFAULT_FONT, 9),
            bg="white", 
            fg="#64748b",
            justify="left"
        )
        self.stats_label.pack()
        
        # Botón finalizar
        finish_btn = tk.Button(
            right_content, 
            text="💾 Guardar y Continuar",
            font=(DEFAULT_FONT, 12, "bold"),
            bg="#10B981", 
            fg="white",
            activebackground="#059669",
            relief="flat", 
            pady=15,
            cursor="hand2",
            command=self.finish_labeling
        )
        finish_btn.pack(fill="x", padx=12, pady=15)
        
        # Atajos de teclado info
        shortcuts_frame = tk.Frame(right_content, bg="#F1F5F9")
        shortcuts_frame.pack(fill="x", padx=12, pady=(0, 10))
        
        shortcuts_title = tk.Label(
            shortcuts_frame,
            text="⌨️ Atajos de Teclado",
            font=(DEFAULT_FONT, 9, "bold"),
            bg="#F1F5F9",
            fg="#475569"
        )
        shortcuts_title.pack(pady=(8, 5))
        
        shortcuts_text = "0-4: Asignar clase\n← →: Navegar\nEspacio: Siguiente"
        shortcuts_label = tk.Label(
            shortcuts_frame,
            text=shortcuts_text,
            font=(DEFAULT_FONT, 8),
            bg="#F1F5F9",
            fg="#64748b",
            justify="center"
        )
        shortcuts_label.pack(pady=(0, 8))
        
        # Configurar el scroll del panel derecho
        def configure_right_scroll(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        
        right_content.bind("<Configure>", configure_right_scroll)
        
        # Habilitar scroll con mouse
        def on_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        right_canvas.bind("<MouseWheel>", on_mousewheel)
    
    def on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        """Reset the canvas window to encompass inner frame when required"""
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def adjust_color_brightness(self, hex_color, factor):
        """Ajustar brillo de un color hexadecimal"""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def load_image(self):
        """Cargar y mostrar la imagen actual"""
        if self.current_index >= len(self.images):
            self.finish_labeling()
            return
        
        image_name = self.images[self.current_index]
        image_path = os.path.join(self.data_folder, image_name)
        
        try:
            # Cargar imagen
            img = Image.open(image_path)
            img.thumbnail((700, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
        except Exception as e:
            print(f"⚠️ Error cargando imagen {image_name}: {e}")
            self.image_label.config(text=f"Error cargando imagen:\n{e}")
        
        # Actualizar información
        labeled_count = len([k for k in self.labels if k in self.images])
        total_count = len(self.images)
        progress_pct = (labeled_count / total_count * 100) if total_count > 0 else 0
        
        self.progress_label.config(
            text=f"Imagen {self.current_index + 1} de {total_count} | "
                 f"Etiquetadas: {labeled_count}/{total_count} ({progress_pct:.1f}%)"
        )
        self.name_label.config(text=f"📄 {image_name}")
        
        # Actualizar estado de etiquetado
        if image_name in self.labels:
            class_id = self.labels[image_name]
            class_name, color, _ = self.class_info[class_id]
            self.current_label_text.config(
                text=f"✓ {class_name}", 
                bg=color, 
                fg="white"
            )
            self.current_label_frame.config(bg=color)
        else:
            self.current_label_text.config(
                text="⚠ Sin etiquetar", 
                bg="#FEF3C7", 
                fg="#92400E"
            )
            self.current_label_frame.config(bg="#FEF3C7")
        
        # Actualizar estadísticas
        self.update_statistics()
    
    def assign_class(self, class_id):
        """Asignar clase a la imagen actual"""
        if self.current_index >= len(self.images):
            return
        
        image_name = self.images[self.current_index]
        self.labels[image_name] = class_id
        self.save_labels()
        
        # Feedback visual
        class_name, color, _ = self.class_info[class_id]
        self.current_label_text.config(
            text=f"✓ {class_name}", 
            bg=color, 
            fg="white"
        )
        self.current_label_frame.config(bg=color)
        
        # Auto-avanzar después de un breve delay
        self.root.after(200, self.next_image)
    
    def next_image(self):
        """Ir a la siguiente imagen"""
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.load_image()
        else:
            messagebox.showinfo(
                "Fin",
                "Has llegado al final de las imágenes.\n\n"
                "Haz clic en 'Guardar y Continuar' para finalizar."
            )
    
    def previous_image(self):
        """Ir a la imagen anterior"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()
    
    def update_statistics(self):
        """Actualizar estadísticas de etiquetado"""
        stats = {}
        for img_name, class_id in self.labels.items():
            if img_name in self.images:
                stats[class_id] = stats.get(class_id, 0) + 1
        
        stats_text = ""
        for class_id in range(5):
            count = stats.get(class_id, 0)
            name = self.class_info[class_id][0].split(' - ')[1]
            stats_text += f"Clase {class_id} ({name}): {count}\n"
        
        self.stats_label.config(text=stats_text)
    
    def finish_labeling(self):
        """Finalizar proceso de etiquetado"""
        labeled = len([k for k in self.labels if k in self.images])
        total = len(self.images)
        
        if labeled == 0:
            messagebox.showwarning(
                "Sin Etiquetas",
                "No has etiquetado ninguna imagen.\n\n"
                "Etiqueta al menos algunas imágenes antes de continuar."
            )
            return
        
        if labeled < 50:
            response = messagebox.askyesno(
                "Pocas Etiquetas",
                f"Has etiquetado solo {labeled} de {total} imágenes.\n\n"
                "Se recomienda etiquetar al menos 50 imágenes para un buen entrenamiento.\n\n"
                "¿Deseas continuar de todas formas?"
            )
            if not response:
                return
        
        # Guardar etiquetas finales
        self.save_labels()
        
        # Mostrar resumen
        summary = f"✅ Etiquetado completado!\n\n"
        summary += f"📊 Total etiquetado: {labeled} de {total} imágenes\n"
        summary += f"📁 Guardado en: {self.labels_file}\n\n"
        
        stats = {}
        for img_name, class_id in self.labels.items():
            if img_name in self.images:
                stats[class_id] = stats.get(class_id, 0) + 1
        
        summary += "Distribución por clase:\n"
        for class_id in sorted(stats.keys()):
            count = stats[class_id]
            name = self.class_info[class_id][0]
            summary += f"  • {name}: {count}\n"
        
        summary += f"\n🎓 Ahora puedes entrenar el modelo SVM\n"
        summary += f"   ejecutando: python main.py"
        
        messagebox.showinfo("Completado", summary)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageLabeler(root)
    root.mainloop()