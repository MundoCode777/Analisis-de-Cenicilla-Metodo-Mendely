import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import os
import platform
import numpy as np
import threading
from modelos.svm_model import SVMModel
from modelos.cnn_model import CNNModel
from modelos.transformer_model import TransformerModel
from modelos.efficientnet_model import EfficientNetModel
from modelos.convnext_model import ConvNeXtModel
from modelos.swin_transformer_model import SwinTransformerModel
from metricas import MetricsEvaluator

# Helper: detectar sistema para tipografías
DEFAULT_FONT = "Segoe UI" if platform.system() == "Windows" else "Helvetica"

class ScrollableImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Cenicilla en Hojas")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.config(bg="#F8FAFF")
        
        # Paleta premium mejorada
        self.colors = {
            'bg': '#F8FAFF',
            'card': '#FFFFFF',
            'primary': '#6366F1',
            'primary_hover': '#4F46E5',
            'success': '#10B981',
            'success_hover': '#059669',
            'warning': '#F59E0B',
            'danger': '#EF4444',
            'text_dark': '#0f172a',
            'text_medium': '#475569',
            'text_light': '#94a3b8',
            'border': '#E0E7FF',
            'shadow': '#C7D2FE',
            'accent': '#8B5CF6',
            'accent_light': '#DDD6FE'
        }
        
        self.image_path = None
        self.models = {}
        self.analysis_in_progress = False
        self.evaluator = MetricsEvaluator()
        
        # Inicializar modelos
        self.init_models()
        self.create_widgets()
    
    def init_models(self):
        """Inicializar los seis modelos"""
        try:
            print("\n" + "="*60)
            print("🤖 INICIALIZANDO MODELOS")
            print("="*60)
            
            self.models['SVM'] = SVMModel()
            print(f"   SVM: {'✅ Entrenado' if hasattr(self.models['SVM'], 'is_trained') and self.models['SVM'].is_trained else '❌ No entrenado'}")
            
            self.models['CNN'] = CNNModel()
            print(f"   CNN: {'✅ Entrenado' if hasattr(self.models['CNN'], 'is_trained') and self.models['CNN'].is_trained else '❌ No entrenado'}")
            
            self.models['Transformer'] = TransformerModel()
            
            transformer_trained = hasattr(self.models['Transformer'], 'is_trained') and self.models['Transformer'].is_trained
            transformer_has_model = hasattr(self.models['Transformer'], 'model') and self.models['Transformer'].model is not None
            
            print(f"   Transformer: {'✅ Entrenado' if transformer_trained else '❌ No entrenado'}")
            print(f"      - is_trained: {transformer_trained}")
            print(f"      - model exists: {transformer_has_model}")
            
            if os.path.exists("modelos/transformer_model.h5"):
                print(f"      - Archivo modelo: ✅ Existe")
            else:
                print(f"      - Archivo modelo: ❌ No existe")

            self.models['EfficientNet'] = EfficientNetModel()
            print(f"   EfficientNet: {'✅ Entrenado' if hasattr(self.models['EfficientNet'], 'is_trained') and self.models['EfficientNet'].is_trained else '❌ No entrenado'}")

            self.models['ConvNeXt'] = ConvNeXtModel()
            print(f"   ConvNeXt: {'✅ Entrenado' if hasattr(self.models['ConvNeXt'], 'is_trained') and self.models['ConvNeXt'].is_trained else '❌ No entrenado'}")

            self.models['Swin'] = SwinTransformerModel()
            print(f"   Swin: {'✅ Entrenado' if hasattr(self.models['Swin'], 'is_trained') and self.models['Swin'].is_trained else '❌ No entrenado'}")
            
            untrained = []
            for name, model in self.models.items():
                if not (hasattr(model, 'is_trained') and model.is_trained):
                    untrained.append(name)
            
            if untrained:
                print(f"\n⚠️ ADVERTENCIA: Los siguientes modelos NO están entrenados:")
                for name in untrained:
                    print(f"   - {name}")
                print(f"\n💡 Para entrenar estos modelos:")
                print(f"   1. Ejecuta: python etiquetador.py")
                print(f"   2. Etiqueta al menos 50 imágenes por modelo")
                print(f"   3. Reinicia la aplicación")
            else:
                print(f"\n✅ Todos los modelos están listos para usar")
            
            print("="*60 + "\n")
            
        except Exception as e:
            import traceback
            print(f"❌ Error inicializando modelos: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Error inicializando modelos: {e}")
    
    def create_widgets(self):
        # Canvas principal con scroll
        self.canvas = tk.Canvas(self.root, bg=self.colors['bg'], highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Scrollbar estilizada
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except:
            pass
        
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Frame principal
        self.main_frame = tk.Frame(self.canvas, bg=self.colors['bg'])
        self.canvas_window = self.canvas.create_window((0,0), window=self.main_frame, anchor="nw")
        self.main_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        
        # Container con padding
        content = tk.Frame(self.main_frame, bg=self.colors['bg'])
        content.pack(expand=True, fill="both", padx=55, pady=35)
        
        # ========== HEADER MEJORADO ==========
        header = tk.Frame(content, bg=self.colors['bg'])
        header.pack(fill="x", pady=(0, 25), anchor="w")
        
        # Fila 1: Logo + Título
        row1 = tk.Frame(header, bg=self.colors['bg'])
        row1.pack(anchor="w", pady=(0, 10))
        
        logo = tk.Label(row1, text="🌿", font=(DEFAULT_FONT, 42), bg=self.colors['bg'])
        logo.pack(side="left", padx=(0, 15))
        
        title = tk.Label(row1, text="Analizador de Cenicilla en Hojas", 
                        font=(DEFAULT_FONT, 32, "bold"),
                        bg=self.colors['bg'], 
                        fg=self.colors['text_dark'])
        title.pack(side="left")
        
        # Fila 2: Descripción
        row2 = tk.Frame(header, bg=self.colors['bg'])
        row2.pack(anchor="w", pady=(0, 12))
        
        desc = tk.Label(row2, 
                       text="Detección inteligente de cenicilla • Análisis multi-modelo • Clasificación por severidad • Evaluación comparativa",
                       font=(DEFAULT_FONT, 12),
                       bg=self.colors['bg'], 
                       fg=self.colors['text_medium'])
        desc.pack(anchor="w")
        
        # Fila 3: Badges de modelos
        row3 = tk.Frame(header, bg=self.colors['bg'])
        row3.pack(anchor="w")
        
        model_badges = [
            ("🤖 SVM", self.colors['primary']),
            ("🧠 CNN", self.colors['accent']),
            ("⚡ Transformer", self.colors['warning']),
            ("🚀 EfficientNet", self.colors['success']),
            ("📐 ConvNeXt", self.colors['danger']),
            ("🪟 Swin", self.colors['primary_hover']),
            ("📊 5 Clases", self.colors['success'])
        ]
        
        for text, color in model_badges:
            badge = tk.Label(row3, text=text, 
                           font=(DEFAULT_FONT, 10, "bold"),
                           bg=color, fg="white", 
                           padx=18, pady=8, relief="flat")
            badge.pack(side="left", padx=(0, 8))
        
        # ========== CARD PRINCIPAL MEJORADA ==========
        card_outer = tk.Frame(content, bg=self.colors['shadow'])
        card_outer.pack(fill="both", expand=True, pady=(0, 5))
        
        self.main_card = tk.Frame(card_outer, 
                                  bg=self.colors['card'],
                                  highlightbackground=self.colors['primary'],
                                  highlightthickness=3,
                                  relief="flat")
        self.main_card.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Barra decorativa superior
        header_bar = tk.Frame(self.main_card, bg=self.colors['card'], height=12)
        header_bar.pack(fill="x")
        
        bar1 = tk.Frame(header_bar, bg=self.colors['primary'], height=12)
        bar1.pack(side="left", fill="both", expand=True)
        
        bar2 = tk.Frame(header_bar, bg=self.colors['accent'], height=12)
        bar2.pack(side="left", fill="both", expand=True)
        
        # Área de imagen
        img_area = tk.Frame(self.main_card, bg=self.colors['card'])
        img_area.pack(fill="both", expand=True, padx=40, pady=35)
        
        # Preview frame con estilo glass
        self.preview_frame = tk.Frame(img_area, 
                                      bg=self.colors['bg'],
                                      highlightbackground=self.colors['border'],
                                      highlightthickness=3,
                                      relief="flat")
        self.preview_frame.pack(fill="both", expand=True)
        
        preview_inner = tk.Frame(self.preview_frame, bg=self.colors['bg'])
        preview_inner.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.preview_label = tk.Label(preview_inner, text="", bg=self.colors['bg'])
        self.preview_label.pack(expand=True, fill="both")
        
        # ========== PLACEHOLDER MEJORADO ==========
        self.placeholder = tk.Frame(preview_inner, bg=self.colors['bg'])
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")
        
        # Frame decorativo con borde
        deco_frame = tk.Frame(self.placeholder, 
                             bg=self.colors['border'],
                             highlightbackground=self.colors['primary'],
                             highlightthickness=2,
                             relief="flat")
        deco_frame.pack(padx=30, pady=30)
        
        # Interior del frame
        inner_frame = tk.Frame(deco_frame, bg=self.colors['bg'])
        inner_frame.pack(padx=50, pady=40)
        
        # Icono principal
        icon = tk.Label(inner_frame, text="🌿", 
                       font=(DEFAULT_FONT, 80),
                       bg=self.colors['bg'])
        icon.pack(pady=(10, 20))
        
        # Texto principal
        text1 = tk.Label(inner_frame, 
                        text="Análisis de Cenicilla en Hojas",
                        font=(DEFAULT_FONT, 16, "bold"),
                        bg=self.colors['bg'], 
                        fg=self.colors['text_dark'])
        text1.pack(pady=(0, 8))
        
        # Subtexto 1
        text2 = tk.Label(inner_frame, 
                        text="Haz clic en 'Analizar Dataset' para procesar todas las imágenes",
                        font=(DEFAULT_FONT, 12),
                        bg=self.colors['bg'], 
                        fg=self.colors['text_medium'])
        text2.pack(pady=(0, 5))
        
        # Subtexto 2
        text3 = tk.Label(inner_frame, 
                        text="o en 'Cargar Imagen' para analizar una imagen individual",
                        font=(DEFAULT_FONT, 12),
                        bg=self.colors['bg'], 
                        fg=self.colors['text_medium'])
        text3.pack(pady=(0, 15))
        
        # Separador
        sep = tk.Frame(inner_frame, bg=self.colors['border'], height=2)
        sep.pack(fill="x", padx=20, pady=10)
        
        # Info de modelos
        text4 = tk.Label(inner_frame, 
                        text="🤖 Se utilizarán 6 modelos: SVM, CNN, Vision Transformer,",
                        font=(DEFAULT_FONT, 11),
                        bg=self.colors['bg'], 
                        fg=self.colors['primary'])
        text4.pack(pady=(5, 0))

        text4b = tk.Label(inner_frame,
                        text="EfficientNet, ConvNeXt y Swin Transformer",
                        font=(DEFAULT_FONT, 11),
                        bg=self.colors['bg'],
                        fg=self.colors['primary'])
        text4b.pack(pady=(0, 0))
        
        # Info de clases
        text5 = tk.Label(inner_frame, 
                        text="📊 Clasificación en 5 niveles de severidad",
                        font=(DEFAULT_FONT, 11),
                        bg=self.colors['bg'], 
                        fg=self.colors['success'])
        text5.pack(pady=(3, 10))
        
        # Info bar
        self.info_bar = tk.Frame(img_area, bg=self.colors['card'], height=60)
        self.info_bar.pack(fill="x", pady=(25, 0))
        
        self.info_label = tk.Label(self.info_bar, text="",
                                   font=(DEFAULT_FONT, 12),
                                   bg=self.colors['card'],
                                   fg=self.colors['text_medium'])
        self.info_label.pack(expand=True, fill="x", pady=10)
        
        # ========== BOTONES MEJORADOS ==========
        btn_section = tk.Frame(self.main_card, bg=self.colors['card'])
        btn_section.pack(fill="x", pady=(0, 30))
        
        btn_container = tk.Frame(btn_section, bg=self.colors['card'])
        btn_container.pack()
        
        # Botón Analizar Dataset
        analyze_shadow = tk.Frame(btn_container, bg=self.colors['shadow'])
        analyze_shadow.pack(side="left", padx=12)
        
        self.btn_analyze = tk.Button(analyze_shadow,
                                     text="🔬 Analizar Dataset",
                                     font=(DEFAULT_FONT, 14, "bold"),
                                     bg=self.colors['primary'],
                                     fg="white",
                                     activebackground=self.colors['primary_hover'],
                                     activeforeground="white",
                                     relief="flat",
                                     padx=35,
                                     pady=16,
                                     cursor="hand2",
                                     command=self.analyze_dataset,
                                     borderwidth=0)
        self.btn_analyze.pack(padx=3, pady=3)
        self.add_hover(self.btn_analyze, self.colors['primary'], self.colors['primary_hover'])
        
        # Botón Cargar Imagen Individual
        upload_shadow = tk.Frame(btn_container, bg=self.colors['shadow'])
        upload_shadow.pack(side="left", padx=12)
        
        self.btn_upload = tk.Button(upload_shadow, 
                                    text="📁 Cargar Imagen",
                                    font=(DEFAULT_FONT, 13, "bold"),
                                    bg=self.colors['success'],
                                    fg="white",
                                    activebackground=self.colors['success_hover'],
                                    activeforeground="white",
                                    relief="flat",
                                    padx=30,
                                    pady=14,
                                    cursor="hand2",
                                    command=self.select_image,
                                    borderwidth=0)
        self.btn_upload.pack(padx=3, pady=3)
        self.add_hover(self.btn_upload, self.colors['success'], self.colors['success_hover'])
        
        # Botón Limpiar
        clear_shadow = tk.Frame(btn_container, bg=self.colors['shadow'])
        clear_shadow.pack(side="left", padx=12)
        
        self.btn_clear = tk.Button(clear_shadow,
                                   text="🧹 Limpiar",
                                   font=(DEFAULT_FONT, 13),
                                   bg=self.colors['border'],
                                   fg=self.colors['text_dark'],
                                   activebackground=self.colors['shadow'],
                                   activeforeground=self.colors['text_dark'],
                                   relief="flat",
                                   padx=25,
                                   pady=14,
                                   cursor="hand2",
                                   command=self.clear_image,
                                   borderwidth=0)
        self.btn_clear.pack(padx=3, pady=3)
        self.add_hover(self.btn_clear, self.colors['border'], self.colors['shadow'])
        
        # Botón Configurar Referencias
        config_shadow = tk.Frame(btn_container, bg=self.colors['shadow'])
        config_shadow.pack(side="left", padx=12)
        
        self.btn_config = tk.Button(config_shadow,
                                    text="⚙️ Configurar Referencias",
                                    font=(DEFAULT_FONT, 13),
                                    bg=self.colors['accent'],
                                    fg="white",
                                    activebackground=self.colors['primary'],
                                    activeforeground="white",
                                    relief="flat",
                                    padx=25,
                                    pady=14,
                                    cursor="hand2",
                                    command=self.configure_references,
                                    borderwidth=0)
        self.btn_config.pack(padx=3, pady=3)
        self.add_hover(self.btn_config, self.colors['accent'], self.colors['primary'])
        
        # Frame para análisis
        self.analysis_frame = tk.Frame(content, bg=self.colors['bg'])
        
        # ========== FOOTER MEJORADO ==========
        footer = tk.Frame(content, bg=self.colors['bg'])
        footer.pack(fill="x", pady=(30, 0))
        
        line = tk.Frame(footer, bg=self.colors['border'], height=2)
        line.pack(fill="x", pady=(0, 15))
        
        footer_text = tk.Label(footer, 
                              text="© 2025 Analizador de Cenicilla • Powered by Multi-Model AI Technology",
                              font=(DEFAULT_FONT, 11),
                              bg=self.colors['bg'],
                              fg=self.colors['text_light'])
        footer_text.pack(pady=5)
        
        version = tk.Label(footer, text="Version 3.0",
                          font=(DEFAULT_FONT, 9),
                          bg=self.colors['bg'],
                          fg=self.colors['text_light'])
        version.pack()
    
    def configure_references(self):
        """Abrir ventana para configurar imágenes de referencia"""
        config_window = tk.Toplevel(self.root)
        config_window.title("Configurar Imágenes de Referencia")
        config_window.geometry("900x700")
        config_window.config(bg=self.colors['bg'])
        config_window.transient(self.root)
        config_window.grab_set()
        
        # Título
        title_frame = tk.Frame(config_window, bg=self.colors['primary'])
        title_frame.pack(fill="x")
        
        title = tk.Label(
            title_frame,
            text="⚙️ CONFIGURAR IMÁGENES DE REFERENCIA",
            font=(DEFAULT_FONT, 18, "bold"),
            bg=self.colors['primary'],
            fg="white"
        )
        title.pack(pady=20)
        
        # Descripción
        desc_frame = tk.Frame(config_window, bg=self.colors['card'])
        desc_frame.pack(fill="x", padx=20, pady=15)
        
        desc = tk.Label(
            desc_frame,
            text="Carga imágenes de referencia para cada clase de cenicilla.\n"
                 "Estas imágenes se mostrarán en los resultados del análisis.",
            font=(DEFAULT_FONT, 11),
            bg=self.colors['card'],
            fg=self.colors['text_medium'],
            justify="center"
        )
        desc.pack(pady=10)
        
        # Crear carpeta referencias si no existe
        if not os.path.exists("referencias"):
            os.makedirs("referencias")
            info = tk.Label(
                desc_frame,
                text="✅ Carpeta 'referencias' creada automáticamente",
                font=(DEFAULT_FONT, 10, "bold"),
                bg=self.colors['card'],
                fg=self.colors['success']
            )
            info.pack(pady=5)
        
        # Canvas con scroll para las clases
        canvas_frame = tk.Frame(config_window, bg=self.colors['bg'])
        canvas_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        canvas = tk.Canvas(canvas_frame, bg=self.colors['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        classes_frame = tk.Frame(canvas, bg=self.colors['bg'])
        
        classes_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=classes_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        class_names = [
            "Resistente",
            "Moderadamente Tolerante",
            "Ligeramente Tolerante",
            "Susceptible",
            "Altamente Susceptible"
        ]
        
        class_colors = {
            1: "#10B981",
            2: "#84CC16",
            3: "#F59E0B",
            4: "#F97316",
            5: "#EF4444"
        }
        
        # Crear tarjeta para cada clase
        self.reference_labels = {}
        
        for class_num in range(1, 6):
            # Frame de la clase
            class_card = tk.Frame(
                classes_frame,
                bg=self.colors['card'],
                highlightbackground=class_colors[class_num],
                highlightthickness=3
            )
            class_card.pack(fill="x", padx=10, pady=10)
            
            # Header de la clase
            header = tk.Frame(class_card, bg=class_colors[class_num])
            header.pack(fill="x")
            
            tk.Label(
                header,
                text=f"CLASE {class_num} - {class_names[class_num-1]}",
                font=(DEFAULT_FONT, 13, "bold"),
                bg=class_colors[class_num],
                fg="white"
            ).pack(pady=12)
            
            # Contenido
            content = tk.Frame(class_card, bg=self.colors['card'])
            content.pack(fill="x", padx=15, pady=15)
            
            # Imagen actual
            img_frame = tk.Frame(content, bg=self.colors['bg'])
            img_frame.pack(side="left", padx=10)
            
            ref_img = self.load_reference_image(class_num)
            ref_photo = ImageTk.PhotoImage(ref_img)
            
            img_label = tk.Label(img_frame, image=ref_photo, bg=self.colors['bg'])
            img_label.image = ref_photo
            img_label.pack()
            
            self.reference_labels[class_num] = img_label
            
            # Info y botones
            info_frame = tk.Frame(content, bg=self.colors['card'])
            info_frame.pack(side="left", fill="both", expand=True, padx=20)
            
            # Estado actual
            reference_exists = False
            reference_filename = ""
            for ext in ['.png', '.jpg', '.jpeg']:
                reference_path = f"referencias/clase_{class_num}{ext}"
                if os.path.exists(reference_path):
                    reference_exists = True
                    reference_filename = f"clase_{class_num}{ext}"
                    break
            
            if reference_exists:
                status_text = f"✅ Imagen configurada: {reference_filename}"
                status_color = self.colors['success']
            else:
                status_text = f"⚠️ No hay imagen de referencia"
                status_color = self.colors['warning']
            
            status_label = tk.Label(
                info_frame,
                text=status_text,
                font=(DEFAULT_FONT, 10, "bold"),
                bg=self.colors['card'],
                fg=status_color
            )
            status_label.pack(anchor="w", pady=(0, 10))
            
            # Descripción
            desc_text = f"Esta clase representa: {class_names[class_num-1].upper()}"
            tk.Label(
                info_frame,
                text=desc_text,
                font=(DEFAULT_FONT, 10),
                bg=self.colors['card'],
                fg=self.colors['text_medium'],
                wraplength=350,
                justify="left"
            ).pack(anchor="w", pady=(0, 15))
            
            # Botón para cargar imagen
            btn_frame = tk.Frame(info_frame, bg=self.colors['card'])
            btn_frame.pack(anchor="w")
            
            load_btn = tk.Button(
                btn_frame,
                text=f"📁 Cargar Imagen para Clase {class_num}",
                font=(DEFAULT_FONT, 11, "bold"),
                bg=class_colors[class_num],
                fg="white",
                padx=20,
                pady=10,
                cursor="hand2",
                relief="flat",
                command=lambda cn=class_num, il=img_label, sl=status_label: 
                    self.load_reference_for_class(cn, il, sl)
            )
            load_btn.pack(side="left", padx=(0, 10))
            
            # Botón para eliminar
            reference_exists_for_delete = False
            for ext in ['.png', '.jpg', '.jpeg']:
                if os.path.exists(f"referencias/clase_{class_num}{ext}"):
                    reference_exists_for_delete = True
                    break
            
            if reference_exists_for_delete:
                delete_btn = tk.Button(
                    btn_frame,
                    text="🗑️ Eliminar",
                    font=(DEFAULT_FONT, 10),
                    bg=self.colors['danger'],
                    fg="white",
                    padx=15,
                    pady=10,
                    cursor="hand2",
                    relief="flat",
                    command=lambda cn=class_num, il=img_label, sl=status_label: 
                        self.delete_reference_for_class(cn, il, sl)
                )
                delete_btn.pack(side="left")
        
        # Botón cerrar
        close_btn = tk.Button(
            config_window,
            text="✅ Cerrar",
            font=(DEFAULT_FONT, 13, "bold"),
            bg=self.colors['success'],
            fg="white",
            padx=40,
            pady=12,
            cursor="hand2",
            relief="flat",
            command=config_window.destroy
        )
        close_btn.pack(pady=20)
    
    def load_reference_for_class(self, class_num, img_label, status_label):
        """Cargar imagen de referencia para una clase específica"""
        path = filedialog.askopenfilename(
            title=f"Seleccionar imagen de referencia para Clase {class_num}",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.gif *.bmp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Todos", "*.*")
            ]
        )
        
        if path:
            try:
                img = Image.open(path)
                _, ext = os.path.splitext(path)
                if ext.lower() not in ['.png', '.jpg', '.jpeg']:
                    ext = '.png'
                
                reference_path = f"referencias/clase_{class_num}{ext.lower()}"
                
                if img.mode in ('RGBA', 'LA', 'P'):
                    if ext.lower() in ['.jpg', '.jpeg']:
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background
                
                if ext.lower() == '.png':
                    img.save(reference_path, "PNG")
                else:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(reference_path, "JPEG", quality=95)
                
                ref_img = self.load_reference_image(class_num)
                ref_photo = ImageTk.PhotoImage(ref_img)
                img_label.config(image=ref_photo)
                img_label.image = ref_photo
                
                status_label.config(
                    text=f"✅ Imagen configurada: clase_{class_num}{ext.lower()}",
                    fg=self.colors['success']
                )
                
                messagebox.showinfo(
                    "Éxito",
                    f"Imagen de referencia para Clase {class_num} cargada correctamente"
                )
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen:\n{e}")
    
    def delete_reference_for_class(self, class_num, img_label, status_label):
        """Eliminar imagen de referencia para una clase"""
        deleted = False
        for ext in ['.png', '.jpg', '.jpeg']:
            reference_path = f"referencias/clase_{class_num}{ext}"
            if os.path.exists(reference_path):
                if messagebox.askyesno(
                    "Confirmar",
                    f"¿Eliminar imagen de referencia para Clase {class_num}?"
                ):
                    try:
                        os.remove(reference_path)
                        deleted = True
                    except Exception as e:
                        messagebox.showerror("Error", f"No se pudo eliminar:\n{e}")
                        return
                break
        
        if deleted:
            ref_img = self.load_reference_image(class_num)
            ref_photo = ImageTk.PhotoImage(ref_img)
            img_label.config(image=ref_photo)
            img_label.image = ref_photo
            
            status_label.config(
                text=f"⚠️ No hay imagen de referencia",
                fg=self.colors['warning']
            )
            
            messagebox.showinfo("Éxito", "Imagen de referencia eliminada")
    
    def reload_model(self, model_name):
        """Recargar un modelo específico"""
        try:
            print(f"\n🔄 Recargando modelo {model_name}...")
            
            if model_name == 'SVM':
                self.models['SVM'] = SVMModel()
            elif model_name == 'CNN':
                self.models['CNN'] = CNNModel()
            elif model_name == 'Transformer':
                self.models['Transformer'] = TransformerModel()
            elif model_name == 'EfficientNet':
                self.models['EfficientNet'] = EfficientNetModel()
            elif model_name == 'ConvNeXt':
                self.models['ConvNeXt'] = ConvNeXtModel()
            elif model_name == 'Swin':
                self.models['Swin'] = SwinTransformerModel()
            
            model = self.models[model_name]
            is_trained = hasattr(model, 'is_trained') and model.is_trained
            
            if is_trained:
                messagebox.showinfo(
                    "Éxito",
                    f"Modelo {model_name} recargado exitosamente.\n\n"
                    f"Haz clic en 'Analizar Dataset' para ver los nuevos resultados."
                )
                print(f"✅ Modelo {model_name} recargado correctamente")
            else:
                messagebox.showwarning(
                    "Advertencia",
                    f"El modelo {model_name} se recargó pero sigue sin estar entrenado.\n\n"
                    f"Verifica que:\n"
                    f"1. Hayas ejecutado 'etiquetador.py'\n"
                    f"2. Hayas etiquetado suficientes imágenes\n"
                    f"3. El archivo del modelo exista en la carpeta 'modelos/'"
                )
                print(f"⚠️ Modelo {model_name} recargado pero no entrenado")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error recargando modelo {model_name}:\n{e}")
            print(f"❌ Error recargando {model_name}: {e}")
    
    def add_hover(self, widget, normal, hover):
        widget.bind("<Enter>", lambda e: widget.config(bg=hover))
        widget.bind("<Leave>", lambda e: widget.config(bg=normal))
    
    def on_frame_configure(self, e=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_configure(self, e):
        self.canvas.itemconfig(self.canvas_window, width=e.width)
    
    def on_mousewheel(self, e):
        if platform.system() == "Windows":
            self.canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        else:
            self.canvas.yview_scroll(int(-1*e.delta), "units")
    
    def select_image(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.gif *.bmp *.webp *.tiff"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("Todos", "*.*")
            ]
        )
        if path:
            self.image_path = path
            self.display_image(path)
    
    def display_image(self, path):
        try:
            self.placeholder.place_forget()
            
            img = Image.open(path)
            original = img.size
            
            img.thumbnail((800, 500), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            self.preview_label.config(image=photo, bg=self.colors['card'])
            self.preview_label.image = photo
            
            name = os.path.basename(path)
            size = os.path.getsize(path) / 1024
            
            info = f"✅ Imagen cargada: {name}  •  {original[0]}×{original[1]} px  •  {size:.1f} KB"
            self.info_label.config(text=info, fg=self.colors['success'], font=(DEFAULT_FONT, 12, "bold"))
            
            self.preview_frame.config(bg=self.colors['card'])
            self.preview_label.config(bg=self.colors['card'])
            
            self.main_frame.update_idletasks()
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar:\n{e}")
    
    def clear_image(self):
        self.image_path = None
        self.preview_label.config(image="", text="")
        self.placeholder.place(relx=0.5, rely=0.5, anchor="center")
        self.info_label.config(text="")
        self.preview_frame.config(bg=self.colors['bg'])
        self.preview_label.config(bg=self.colors['bg'])
        
        for w in self.analysis_frame.winfo_children():
            w.destroy()
        self.analysis_frame.pack_forget()
        
        self.main_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def analyze_dataset(self):
        """Analizar todas las imágenes con los seis modelos"""
        if not os.path.exists("data"):
            messagebox.showerror("Error", "Carpeta 'data' no encontrada.\n\nCrea una carpeta llamada 'data' y coloca allí las imágenes de hojas.")
            return
        
        image_files = [f for f in os.listdir("data") 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            messagebox.showwarning("Advertencia", "No se encontraron imágenes en la carpeta 'data'.")
            return
        
        self.btn_analyze.config(state="disabled", text="🔄 Analizando...")
        
        self.info_label.config(text=f"🔍 Analizando {len(image_files)} imágenes con 6 modelos...", 
                              fg=self.colors['primary'])
        
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
    
    def run_analysis(self):
        """Ejecutar análisis con los seis modelos"""
        try:
            self.root.after(0, self.clear_analysis_frame)
            
            all_results = {}
            
            for model_name, model in self.models.items():
                print(f"🔍 Ejecutando {model_name}...")
                results = model.analyze_dataset("data")
                all_results[model_name] = results
            
            self.root.after(0, lambda: self.display_results(all_results))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error en análisis: {e}"))
        finally:
            self.root.after(0, lambda: self.btn_analyze.config(state="normal", text="🔬 Analizar Dataset"))
    
    def clear_analysis_frame(self):
        """Limpiar frame de análisis"""
        for widget in self.analysis_frame.winfo_children():
            widget.destroy()
        self.analysis_frame.pack(fill="both", expand=True, pady=20)
    
    def load_reference_image(self, class_num):
        """Cargar imagen de referencia real para una clase"""
        for ext in ['.png', '.jpg', '.jpeg']:
            reference_path = f"referencias/clase_{class_num}{ext}"
            try:
                if os.path.exists(reference_path):
                    img = Image.open(reference_path)
                    if img.mode in ('RGBA', 'LA', 'P'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background
                    img = img.resize((150, 120), Image.Resampling.LANCZOS)
                    return img
            except Exception as e:
                print(f"Error cargando {reference_path}: {e}")
                continue
        
        return self.create_placeholder_reference(class_num)
    
    def create_placeholder_reference(self, class_num):
        """Crear placeholder cuando no hay imagen de referencia"""
        width, height = 150, 120
        img = Image.new('RGB', (width, height), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([0, 0, width-1, height-1], outline=(200, 200, 200), width=2)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        text1 = f"Clase {class_num}"
        text2 = "Sin imagen"
        text3 = "de referencia"
        
        bbox1 = draw.textbbox((0, 0), text1, font=font)
        bbox2 = draw.textbbox((0, 0), text2, font=font)
        bbox3 = draw.textbbox((0, 0), text3, font=font)
        
        x1 = (width - (bbox1[2] - bbox1[0])) // 2
        x2 = (width - (bbox2[2] - bbox2[0])) // 2
        x3 = (width - (bbox3[2] - bbox3[0])) // 2
        
        draw.text((x1, 30), text1, fill=(100, 100, 100), font=font)
        draw.text((x2, 55), text2, fill=(150, 150, 150), font=font)
        draw.text((x3, 75), text3, fill=(150, 150, 150), font=font)
        
        return img
    
    def display_results(self, all_results):
        """Mostrar resultados de los seis modelos con referencias visuales"""
        # ============================================================
        # Validar resultados de modelos no entrenados
        # ============================================================
        for model_name in list(all_results.keys()):
            results = all_results[model_name]
            if results and len(results) > 0:
                has_invalid = any(r.get('class', -1) == -1 for r in results)
                if has_invalid:
                    print(f"⚠️ {model_name} tiene resultados inválidos (modelo no entrenado)")

        # Crear pestañas para cada modelo
        notebook = ttk.Notebook(self.analysis_frame)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Colores para las clases
        class_colors = {
            1: "#10B981",
            2: "#84CC16",
            3: "#F59E0B",
            4: "#F97316",
            5: "#EF4444",
            -1: "#94a3b8"  # Gris para modelos no entrenados
        }

        class_names = [
            "Resistente",
            "Moderadamente Tolerante",
            "Ligeramente Tolerante",
            "Susceptible",
            "Altamente Susceptible"
        ]

        for model_name, results in all_results.items():
            model_trained = getattr(self.models.get(model_name), "is_trained", False)

            outer_frame = tk.Frame(notebook, bg=self.colors['bg'])
            notebook.add(outer_frame, text=f"{model_name}")

            canvas_tab = tk.Canvas(outer_frame, bg=self.colors['bg'], highlightthickness=0)
            scrollbar_tab = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas_tab.yview)
            model_frame = tk.Frame(canvas_tab, bg=self.colors['bg'])

            model_frame.bind(
                "<Configure>",
                lambda e, c=canvas_tab: c.configure(scrollregion=c.bbox("all"))
            )

            canvas_tab.create_window((0, 0), window=model_frame, anchor="nw")
            canvas_tab.configure(yscrollcommand=scrollbar_tab.set)

            canvas_tab.pack(side="left", fill="both", expand=True)
            scrollbar_tab.pack(side="right", fill="y")

            if not model_trained:
                warning_frame = tk.Frame(model_frame, bg=self.colors['card'],
                                         highlightbackground=self.colors['warning'],
                                         highlightthickness=3)
                warning_frame.pack(fill="both", expand=True, padx=20, pady=20)

                icon_label = tk.Label(
                    warning_frame,
                    text="⚠️",
                    font=(DEFAULT_FONT, 80),
                    bg=self.colors['card'],
                    fg=self.colors['warning']
                )
                icon_label.pack(pady=(40, 20))

                title_label = tk.Label(
                    warning_frame,
                    text=f"MODELO {model_name} NO ENTRENADO",
                    font=(DEFAULT_FONT, 20, "bold"),
                    bg=self.colors['card'],
                    fg=self.colors['text_dark']
                )
                title_label.pack(pady=(0, 15))

                message = tk.Label(
                    warning_frame,
                    text=f"El modelo {model_name} necesita ser entrenado antes de poder realizar predicciones.\n\n"
                         f"Para entrenar este modelo:\n\n"
                         f"1. Cierra esta aplicación\n"
                         f"2. Ejecuta: python etiquetador.py\n"
                         f"3. Etiqueta al menos 50 imágenes\n"
                         f"4. Vuelve a abrir esta aplicación\n\n"
                         f"Una vez entrenado, podrás ver los resultados aquí.",
                    font=(DEFAULT_FONT, 12),
                    bg=self.colors['card'],
                    fg=self.colors['text_medium'],
                    justify="center"
                )
                message.pack(pady=(0, 40), padx=40)

                btn_container_warning = tk.Frame(warning_frame, bg=self.colors['card'])
                btn_container_warning.pack(pady=(0, 40))

                info_btn = tk.Button(
                    btn_container_warning,
                    text="📖 Más Información",
                    font=(DEFAULT_FONT, 12, "bold"),
                    bg=self.colors['primary'],
                    fg="white",
                    padx=30,
                    pady=12,
                    cursor="hand2",
                    relief="flat",
                    command=lambda mn=model_name: messagebox.showinfo(
                        f"Información - {mn}",
                        f"El modelo {mn} es un modelo de Machine Learning que requiere entrenamiento previo.\n\n"
                        f"Características del {mn}:\n"
                        f"- Requiere datos etiquetados para aprender\n"
                        f"- Mínimo 50 imágenes etiquetadas recomendadas\n"
                        f"- Mayor precisión con más datos\n\n"
                        f"Usa el script 'etiquetador.py' para etiquetar tus imágenes y entrenar el modelo."
                    )
                )
                info_btn.pack(side="left", padx=5)

                reload_btn = tk.Button(
                    btn_container_warning,
                    text="🔄 Recargar Modelo",
                    font=(DEFAULT_FONT, 12, "bold"),
                    bg=self.colors['success'],
                    fg="white",
                    padx=30,
                    pady=12,
                    cursor="hand2",
                    relief="flat",
                    command=lambda mn=model_name: self.reload_model(mn)
                )
                reload_btn.pack(side="left", padx=5)

                continue
            
            # Título del modelo
            title_frame = tk.Frame(model_frame, bg=self.colors['card'])
            title_frame.pack(fill="x", padx=20, pady=(15, 10))
            
            title_label = tk.Label(
                title_frame,
                text=f"🔬 RESULTADOS DEL MODELO {model_name}",
                font=(DEFAULT_FONT, 16, "bold"),
                bg=self.colors['card'],
                fg=self.colors['text_dark']
            )
            title_label.pack(pady=10)
            
            # ========== IMÁGENES DE REFERENCIA DE LAS 5 CLASES ==========
            reference_frame = tk.Frame(model_frame, bg=self.colors['card'], 
                                      highlightbackground=self.colors['primary'],
                                      highlightthickness=2)
            reference_frame.pack(fill="x", padx=20, pady=15)
            
            ref_title = tk.Label(
                reference_frame,
                text=f"🌿 GUÍA VISUAL - 5 CLASES DE CENICILLA",
                font=(DEFAULT_FONT, 14, "bold"),
                bg=self.colors['card'],
                fg=self.colors['primary']
            )
            ref_title.pack(pady=(15, 10))
            
            ref_subtitle = tk.Label(
                reference_frame,
                text="Referencia para identificar el nivel de severidad de la enfermedad",
                font=(DEFAULT_FONT, 10),
                bg=self.colors['card'],
                fg=self.colors['text_medium']
            )
            ref_subtitle.pack(pady=(0, 15))
            
            classes_container = tk.Frame(reference_frame, bg=self.colors['card'])
            classes_container.pack(fill="x", padx=20, pady=10)
            
            for class_num in range(1, 6):
                class_frame = tk.Frame(classes_container, 
                                      bg="white",
                                      highlightbackground=class_colors[class_num],
                                      highlightthickness=3)
                class_frame.pack(side="left", padx=8, pady=10, expand=True)
                
                class_number = tk.Label(
                    class_frame,
                    text=f"CLASE {class_num}",
                    font=(DEFAULT_FONT, 11, "bold"),
                    bg=class_colors[class_num],
                    fg="white",
                    padx=10,
                    pady=5
                )
                class_number.pack(fill="x")
                
                ref_img = self.load_reference_image(class_num)
                ref_photo = ImageTk.PhotoImage(ref_img)
                
                img_label = tk.Label(class_frame, image=ref_photo, bg="white")
                img_label.image = ref_photo
                img_label.pack(pady=10)
                
                name_label = tk.Label(
                    class_frame,
                    text=class_names[class_num-1],
                    font=(DEFAULT_FONT, 9, "bold"),
                    bg="white",
                    fg=self.colors['text_dark'],
                    wraplength=120
                )
                name_label.pack(pady=(0, 10))
            
            # ========== ESTADÍSTICAS - SOLO CLASE PREDOMINANTE ==========
            stats_frame = tk.Frame(model_frame, bg=self.colors['card'],
                                  highlightbackground=self.colors['border'],
                                  highlightthickness=1)
            stats_frame.pack(fill="x", padx=20, pady=15)
            
            stats_title = tk.Label(
                stats_frame,
                text="📊 ESTADÍSTICAS DEL ANÁLISIS",
                font=(DEFAULT_FONT, 13, "bold"),
                bg=self.colors['card'],
                fg=self.colors['text_dark']
            )
            stats_title.pack(pady=(10, 5))
            
            total_images = len(results)
            class_distribution = {}
            
            for result in results:
                class_num = result['class']
                class_distribution[class_num] = class_distribution.get(class_num, 0) + 1
            
            stats_grid = tk.Frame(stats_frame, bg=self.colors['card'])
            stats_grid.pack(fill="x", padx=20, pady=15)
            
            total_frame = tk.Frame(stats_grid, bg=self.colors['card'])
            total_frame.pack(fill="x", pady=5)
            
            tk.Label(
                total_frame,
                text=f"📷 Total de imágenes analizadas:",
                font=(DEFAULT_FONT, 11),
                bg=self.colors['card'],
                fg=self.colors['text_medium']
            ).pack(side="left")
            
            tk.Label(
                total_frame,
                text=f"{total_images}",
                font=(DEFAULT_FONT, 11, "bold"),
                bg=self.colors['card'],
                fg=self.colors['primary']
            ).pack(side="left", padx=10)
            
            tk.Frame(stats_grid, bg=self.colors['border'], height=1).pack(fill="x", pady=10)
            
            dist_label = tk.Label(
                stats_grid,
                text="📈 Distribución por clase:",
                font=(DEFAULT_FONT, 11, "bold"),
                bg=self.colors['card'],
                fg=self.colors['text_dark']
            )
            dist_label.pack(anchor="w", pady=(5, 10))
            
            if class_distribution:
                predominant_class = max(class_distribution, key=class_distribution.get)
                predominant_count = class_distribution[predominant_class]
                predominant_percentage = (predominant_count / total_images * 100) if total_images > 0 else 0
                
                class_row = tk.Frame(stats_grid, bg=self.colors['card'])
                class_row.pack(fill="x", pady=5)
                
                class_label = tk.Label(
                    class_row,
                    text=f"Clase Predominante: {predominant_class} - {class_names[predominant_class-1]}",
                    font=(DEFAULT_FONT, 11, "bold"),
                    bg=self.colors['card'],
                    fg=class_colors[predominant_class],
                    width=40,
                    anchor="w"
                )
                class_label.pack(side="left", padx=(0, 10))
                
                progress_bg = tk.Frame(class_row, bg=self.colors['border'], height=30, width=300)
                progress_bg.pack(side="left", padx=(0, 10))
                
                if predominant_percentage > 0:
                    progress_fill = tk.Frame(
                        progress_bg,
                        bg=class_colors[predominant_class],
                        height=30,
                        width=int(300 * predominant_percentage / 100)
                    )
                    progress_fill.place(x=0, y=0)
                
                percentage_label = tk.Label(
                    class_row,
                    text=f"{predominant_count} ({predominant_percentage:.1f}%)",
                    font=(DEFAULT_FONT, 12, "bold"),
                    bg=self.colors['card'],
                    fg=class_colors[predominant_class]
                )
                percentage_label.pack(side="left")
            
            # ========== MÉTRICAS DE EVALUACIÓN (SIN TABLA DETALLADA) ==========
            metrics_section = tk.Frame(model_frame, bg=self.colors['card'],
                                      highlightbackground=self.colors['accent'],
                                      highlightthickness=2)
            metrics_section.pack(fill="x", padx=20, pady=15)
            
            metrics_title = tk.Label(
                metrics_section,
                text="📊 MÉTRICAS DE EVALUACIÓN DEL MODELO",
                font=(DEFAULT_FONT, 13, "bold"),
                bg=self.colors['card'],
                fg=self.colors['text_dark']
            )
            metrics_title.pack(pady=(10, 5))
            
            model_metrics = self.evaluator.calculate_metrics(results, model_name)
            
            if 'error' in model_metrics:
                error_label = tk.Label(
                    metrics_section,
                    text=f"⚠️ {model_metrics['error']}\n{model_metrics.get('message', '')}",
                    font=(DEFAULT_FONT, 11),
                    bg=self.colors['card'],
                    fg=self.colors['warning'],
                    justify="center"
                )
                error_label.pack(pady=15)
            else:
                metrics_grid = tk.Frame(metrics_section, bg=self.colors['card'])
                metrics_grid.pack(fill="x", padx=20, pady=15)
                
                main_metrics = [
                    ("🎯 Exactitud", model_metrics['accuracy'], self.colors['primary']),
                    ("🔍 Precisión", model_metrics['precision_weighted'], self.colors['success']),
                    ("📈 Recall", model_metrics['recall_weighted'], self.colors['accent']),
                    ("⭐ F1-Score", model_metrics['f1_weighted'], self.colors['warning'])
                ]
                
                for metric_label, metric_value, metric_color in main_metrics:
                    metric_card = tk.Frame(metrics_grid, bg=self.colors['bg'],
                                          highlightbackground=metric_color,
                                          highlightthickness=2)
                    metric_card.pack(side="left", expand=True, padx=5, pady=5)
                    
                    tk.Label(
                        metric_card,
                        text=metric_label,
                        font=(DEFAULT_FONT, 10),
                        bg=self.colors['bg'],
                        fg=self.colors['text_medium']
                    ).pack(pady=(10, 5))
                    
                    tk.Label(
                        metric_card,
                        text=f"{metric_value:.2%}",
                        font=(DEFAULT_FONT, 20, "bold"),
                        bg=self.colors['bg'],
                        fg=metric_color
                    ).pack(pady=(0, 10), padx=20)
                
                # Matriz de confusión
                cm_frame = tk.Frame(metrics_section, bg=self.colors['bg'])
                cm_frame.pack(fill="x", padx=20, pady=10)
                
                cm_title = tk.Label(
                    cm_frame,
                    text="🔍 Matriz de Confusión",
                    font=(DEFAULT_FONT, 11, "bold"),
                    bg=self.colors['bg'],
                    fg=self.colors['text_dark']
                )
                cm_title.pack(anchor="w", pady=(5, 10))
                
                cm_container = tk.Frame(cm_frame, bg=self.colors['card'])
                cm_container.pack(fill="x")
                
                cm = np.array(model_metrics['confusion_matrix'])
                n_classes = len(cm)
                
                cm_grid = tk.Frame(cm_container, bg=self.colors['card'])
                cm_grid.pack(pady=10, padx=10)
                
                tk.Label(
                    cm_grid,
                    text="Predicho →",
                    font=(DEFAULT_FONT, 9, "bold"),
                    bg=self.colors['card'],
                    fg=self.colors['text_medium']
                ).grid(row=0, column=0, padx=5, pady=5, sticky="e")
                
                for i in range(n_classes):
                    tk.Label(
                        cm_grid,
                        text=f"C{i+1}",
                        font=(DEFAULT_FONT, 9, "bold"),
                        bg=self.colors['accent_light'],
                        fg=self.colors['text_dark'],
                        width=6,
                        height=2
                    ).grid(row=0, column=i+1, padx=2, pady=2)
                
                tk.Label(
                    cm_grid,
                    text="Real\n↓",
                    font=(DEFAULT_FONT, 9, "bold"),
                    bg=self.colors['card'],
                    fg=self.colors['text_medium']
                ).grid(row=1, column=0, rowspan=n_classes, padx=5, pady=5, sticky="n")
                
                for i in range(n_classes):
                    tk.Label(
                        cm_grid,
                        text=f"C{i+1}",
                        font=(DEFAULT_FONT, 9, "bold"),
                        bg=self.colors['accent_light'],
                        fg=self.colors['text_dark'],
                        width=6,
                        height=2
                    ).grid(row=i+1, column=0, padx=2, pady=2, sticky="w")
                    
                    for j in range(n_classes):
                        value = cm[i][j]
                        if i == j:
                            bg_color = self.colors['success'] if value > 0 else self.colors['bg']
                            fg_color = "white" if value > 0 else self.colors['text_medium']
                        else:
                            bg_color = self.colors['danger'] if value > 0 else self.colors['bg']
                            fg_color = "white" if value > 0 else self.colors['text_medium']
                        
                        tk.Label(
                            cm_grid,
                            text=str(value),
                            font=(DEFAULT_FONT, 10, "bold" if value > 0 else "normal"),
                            bg=bg_color,
                            fg=fg_color,
                            width=6,
                            height=2,
                            relief="solid",
                            borderwidth=1
                        ).grid(row=i+1, column=j+1, padx=2, pady=2)
            
            # ========== TABLA DE RESULTADOS DETALLADOS ==========
            table_frame = tk.Frame(model_frame, bg=self.colors['card'],
                                  highlightbackground=self.colors['border'],
                                  highlightthickness=1)
            table_frame.pack(fill="both", expand=True, padx=20, pady=15)
            
            table_title = tk.Label(
                table_frame,
                text="📋 RESULTADOS DETALLADOS",
                font=(DEFAULT_FONT, 13, "bold"),
                bg=self.colors['card'],
                fg=self.colors['text_dark']
            )
            table_title.pack(pady=(10, 5))
            
            tree_container = tk.Frame(table_frame, bg=self.colors['card'])
            tree_container.pack(fill="both", expand=True, padx=10, pady=10)
            
            tree = ttk.Treeview(
                tree_container,
                columns=('Imagen', 'Clase', 'Nivel', 'Confianza', 'Estado'),
                show='headings',
                height=15
            )
            
            tree.heading('Imagen', text='📷 Imagen')
            tree.heading('Clase', text='🔢 Clase')
            tree.heading('Nivel', text='🌿 Nivel de Severidad')
            tree.heading('Confianza', text='📈 Confianza')
            tree.heading('Estado', text='⚠️ Estado')
            
            tree.column('Imagen', width=200, anchor="w")
            tree.column('Clase', width=80, anchor="center")
            tree.column('Nivel', width=250, anchor="w")
            tree.column('Confianza', width=120, anchor="center")
            tree.column('Estado', width=150, anchor="center")
            
            tree_scrollbar = ttk.Scrollbar(tree_container, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=tree_scrollbar.set)
            tree_scrollbar.pack(side="right", fill="y")
            tree.pack(side="left", fill="both", expand=True)
            
            for idx, result in enumerate(results):
                class_num = result['class']
                confidence = result['confidence']
                
                if confidence > 0.8:
                    estado = "✅ Alta"
                elif confidence > 0.6:
                    estado = "⚠️ Media"
                else:
                    estado = "❌ Baja"
                
                item_id = tree.insert('', 'end', values=(
                    result['image_name'],
                    f"Clase {class_num}",
                    class_names[class_num-1],
                    f"{confidence:.2%}",
                    estado
                ))
                
                if idx % 2 == 0:
                    tree.item(item_id, tags=('evenrow',))
                else:
                    tree.item(item_id, tags=('oddrow',))
            
            tree.tag_configure('evenrow', background='#F8FAFF')
            tree.tag_configure('oddrow', background='#FFFFFF')
            
            # ========== RESUMEN DE CONFIANZA ==========
            confidence_frame = tk.Frame(model_frame, bg=self.colors['card'],
                                       highlightbackground=self.colors['success'],
                                       highlightthickness=2)
            confidence_frame.pack(fill="x", padx=20, pady=15)
            
            conf_title = tk.Label(
                confidence_frame,
                text="💯 ANÁLISIS DE CONFIANZA",
                font=(DEFAULT_FONT, 13, "bold"),
                bg=self.colors['card'],
                fg=self.colors['text_dark']
            )
            conf_title.pack(pady=(10, 5))
            
            confidences = [r['confidence'] for r in results]
            avg_confidence = np.mean(confidences) if confidences else 0
            max_confidence = np.max(confidences) if confidences else 0
            min_confidence = np.min(confidences) if confidences else 0
            
            conf_grid = tk.Frame(confidence_frame, bg=self.colors['card'])
            conf_grid.pack(fill="x", padx=20, pady=15)
            
            metrics = [
                ("📊 Confianza Promedio", f"{avg_confidence:.2%}", self.colors['primary']),
                ("📈 Confianza Máxima", f"{max_confidence:.2%}", self.colors['success']),
                ("📉 Confianza Mínima", f"{min_confidence:.2%}", self.colors['warning'])
            ]
            
            for label_text, value_text, color in metrics:
                metric_frame = tk.Frame(conf_grid, bg=self.colors['card'])
                metric_frame.pack(side="left", expand=True, padx=10)
                
                tk.Label(
                    metric_frame,
                    text=label_text,
                    font=(DEFAULT_FONT, 10),
                    bg=self.colors['card'],
                    fg=self.colors['text_medium']
                ).pack()
                
                tk.Label(
                    metric_frame,
                    text=value_text,
                    font=(DEFAULT_FONT, 18, "bold"),
                    bg=self.colors['card'],
                    fg=color
                ).pack(pady=5)
        
        # ========== COMPARATIVA ENTRE MODELOS ==========
        self.create_comparison_chart(all_results)
        
        total_images = len(next(iter(all_results.values())))
        self.info_label.config(
            text=f"✅ Análisis completado: {total_images} imágenes procesadas con 6 modelos", 
            fg=self.colors['success'],
            font=(DEFAULT_FONT, 12, "bold")
        )
        
        self.main_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(1.0)
    
    def create_comparison_chart(self, all_results):
        """Crear gráfica comparativa entre modelos"""
        comparison_frame = tk.Frame(self.analysis_frame, bg=self.colors['card'],
                                   highlightbackground=self.colors['accent'],
                                   highlightthickness=3)
        comparison_frame.pack(fill="x", padx=20, pady=20)
        
        title = tk.Label(
            comparison_frame,
            text="📊 COMPARATIVA ENTRE MODELOS",
            font=(DEFAULT_FONT, 16, "bold"),
            bg=self.colors['card'],
            fg=self.colors['text_dark']
        )
        title.pack(pady=(15, 10))
        
        subtitle = tk.Label(
            comparison_frame,
            text="Análisis comparativo del rendimiento de los seis modelos de IA",
            font=(DEFAULT_FONT, 11),
            bg=self.colors['card'],
            fg=self.colors['text_medium']
        )
        subtitle.pack(pady=(0, 15))
        
        tk.Frame(comparison_frame, bg=self.colors['border'], height=2).pack(fill="x", padx=20, pady=10)
        
        trained_models = {name: results for name, results in all_results.items() 
                         if self.models[name].is_trained and results and len(results) > 0 
                         and results[0].get('class', -1) >= 0}
        
        if not trained_models:
            warning_label = tk.Label(
                comparison_frame,
                text="⚠️ No hay modelos entrenados para comparar\n\n"
                     "Entrena al menos un modelo usando 'etiquetador.py' para ver comparativas",
                font=(DEFAULT_FONT, 12, "bold"),
                bg=self.colors['card'],
                fg=self.colors['warning'],
                justify="center"
            )
            warning_label.pack(pady=30)
            return
        
        comparison_grid = tk.Frame(comparison_frame, bg=self.colors['card'])
        comparison_grid.pack(fill="x", padx=30, pady=15)
        
        headers = ["Modelo", "Estado", "Exactitud", "Precisión", "Recall", "F1-Score", "Conf. Promedio"]
        for col, header in enumerate(headers):
            tk.Label(
                comparison_grid,
                text=header,
                font=(DEFAULT_FONT, 11, "bold"),
                bg=self.colors['accent_light'],
                fg=self.colors['text_dark'],
                padx=10,
                pady=8
            ).grid(row=0, column=col, sticky="ew", padx=2, pady=2)
        
        model_colors = {
            'SVM': self.colors['primary'],
            'CNN': self.colors['accent'],
            'Transformer': self.colors['warning'],
            'EfficientNet': self.colors['success'],
            'ConvNeXt': self.colors['danger'],
            'Swin': self.colors['primary_hover'],
        }
        
        row = 1
        for model_name, results in all_results.items():
            is_trained = self.models[model_name].is_trained
            
            model_metrics = self.evaluator.calculate_metrics(results, model_name)
            
            if is_trained and 'error' not in model_metrics:
                estado = "✅ Activo"
                estado_color = self.colors['success']
                exactitud = f"{model_metrics['accuracy']:.2%}"
                precision = f"{model_metrics['precision_weighted']:.2%}"
                recall = f"{model_metrics['recall_weighted']:.2%}"
                f1_score = f"{model_metrics['f1_weighted']:.2%}"
                conf_promedio = f"{model_metrics['avg_confidence']:.2%}"
            else:
                estado = "❌ No entrenado"
                estado_color = self.colors['danger']
                exactitud = "N/A"
                precision = "N/A"
                recall = "N/A"
                f1_score = "N/A"
                conf_promedio = "N/A"
            
            data = [
                (f"🤖 {model_name}", model_colors.get(model_name, self.colors['text_dark']), "white"),
                (estado, self.colors['card'], estado_color),
                (exactitud, self.colors['card'], self.colors['text_dark']),
                (precision, self.colors['card'], self.colors['text_dark']),
                (recall, self.colors['card'], self.colors['text_dark']),
                (f1_score, self.colors['card'], self.colors['text_dark']),
                (conf_promedio, self.colors['card'], self.colors['text_dark'])
            ]
            
            for col, (value, bg_color, fg_color) in enumerate(data):
                tk.Label(
                    comparison_grid,
                    text=value,
                    font=(DEFAULT_FONT, 10, "bold" if col == 0 else "normal"),
                    bg=bg_color,
                    fg=fg_color,
                    padx=10,
                    pady=8
                ).grid(row=row, column=col, sticky="ew", padx=2, pady=2)
            
            row += 1
        
        for col in range(len(headers)):
            comparison_grid.columnconfigure(col, weight=1)
        
        tk.Frame(comparison_frame, bg=self.colors['border'], height=2).pack(fill="x", padx=20, pady=15)
        
        all_metrics = {}
        for model_name, results in all_results.items():
            all_metrics[model_name] = self.evaluator.calculate_metrics(results, model_name)
        
        recommendation_text = self.evaluator.get_model_recommendation(all_metrics)
        
        recommendation_frame = tk.Frame(comparison_frame, bg=self.colors['accent_light'])
        recommendation_frame.pack(fill="x", padx=20, pady=15)
        
        recommendation_label = tk.Label(
            recommendation_frame,
            text=recommendation_text,
            font=(DEFAULT_FONT, 11),
            bg=self.colors['accent_light'],
            fg=self.colors['text_dark'],
            wraplength=1000,
            justify="center"
        )
        recommendation_label.pack(pady=15)
        
        if trained_models:
            export_btn = tk.Button(
                comparison_frame,
                text="📥 Exportar Resultados",
                font=(DEFAULT_FONT, 12, "bold"),
                bg=self.colors['success'],
                fg="white",
                padx=30,
                pady=12,
                cursor="hand2",
                relief="flat",
                command=lambda: self.export_results(all_results)
            )
            export_btn.pack(pady=15)
            self.add_hover(export_btn, self.colors['success'], self.colors['success_hover'])
    
    def export_results(self, all_results):
        """Exportar resultados a un archivo de texto"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Archivo de texto", "*.txt"), ("Todos los archivos", "*.*")]
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("RESULTADOS DEL ANÁLISIS DE CENICILLA EN HOJAS\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for model_name, results in all_results.items():
                        f.write(f"\n{'=' * 80}\n")
                        f.write(f"MODELO: {model_name}\n")
                        f.write(f"{'=' * 80}\n\n")
                        
                        metrics = self.evaluator.calculate_metrics(results, model_name)
                        
                        if 'error' not in metrics:
                            f.write(f"MÉTRICAS DEL MODELO:\n")
                            f.write(f"- Exactitud: {metrics['accuracy']:.2%}\n")
                            f.write(f"- Precisión: {metrics['precision_weighted']:.2%}\n")
                            f.write(f"- Recall: {metrics['recall_weighted']:.2%}\n")
                            f.write(f"- F1-Score: {metrics['f1_weighted']:.2%}\n")
                            f.write(f"- Confianza Promedio: {metrics['avg_confidence']:.2%}\n\n")
                        
                        for result in results:
                            f.write(f"Imagen: {result['image_name']}\n")
                            f.write(f"  - Clase: {result['class']}\n")
                            f.write(f"  - Confianza: {result['confidence']:.2%}\n\n")
                
                messagebox.showinfo("Éxito", f"Resultados exportados correctamente a:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar los resultados:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ScrollableImageApp(root)
    root.mainloop()