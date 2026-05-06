import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk

class VistaPrincipal(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        self.title("Clasificador de Vectores")
        self.geometry("1200x750")
        self.offset = 45 

        self._crear_interfaz()

    def _crear_interfaz(self):

        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        ctk.CTkLabel(self.sidebar, text="CONTROLES", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 20))

        self.btn_subir = ctk.CTkButton(self.sidebar, text="Subir Imagen")
        self.btn_subir.pack(pady=10, padx=20)

        self.btn_cargar_dataset = ctk.CTkButton(self.sidebar, text="Cargar Dataset (K-Means)", fg_color="#E67E22", hover_color="#D35400") # Le pongo un color naranja para diferenciarlo
        self.btn_cargar_dataset.pack(pady=10, padx=20)
        ctk.CTkLabel(self.sidebar, text="Valor de K:", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 0))
        
        self.entry_k = ctk.CTkEntry(self.sidebar, justify="center", width=80)
        self.entry_k.insert(0, "3") # Por defecto sugerimos 3 (cielo, vegetación, agua)
        self.entry_k.pack(pady=5)

        self.btn_entrenar_kmeans = ctk.CTkButton(self.sidebar, text="Entrenar K-Means", fg_color="#27AE60", hover_color="#1E8449")
        self.btn_entrenar_kmeans.pack(pady=10, padx=20)

        self.btn_segmentar_kmeans = ctk.CTkButton(self.sidebar, text="Segmentar Imagen Actual", fg_color="#8E44AD", hover_color="#732D91")
        self.btn_segmentar_kmeans.pack(pady=10, padx=20)

        self.btn_franjas = ctk.CTkButton(self.sidebar, text="Marcar Area", state="disabled")
        self.btn_franjas.pack(pady=10, padx=20)

        ctk.CTkLabel(self.sidebar, text="Elementos por clase:").pack(pady=(20, 0))
        self.entry_elementos = ctk.CTkEntry(self.sidebar, justify="center")
        self.entry_elementos.insert(0, "500")
        self.entry_elementos.pack(pady=5, padx=20)

        self.btn_generar = ctk.CTkButton(self.sidebar, text="Generar Puntos", state="disabled", fg_color="#28a745", hover_color="#218838")
        self.btn_generar.pack(pady=20, padx=20)

        self.btn_evaluar = ctk.CTkButton(self.sidebar, text="Evaluar Clasificador", state="disabled")
        self.btn_evaluar.pack(pady=10, padx=20)

        self.lbl_resultado_eqn = ctk.CTkLabel(self.sidebar, text="Ecuación: ...", font=ctk.CTkFont(size=11), text_color="gray")
        self.lbl_resultado_eqn.pack(side=tk.BOTTOM, pady=20, padx=10)
        
        self.btn_perceptron = ctk.CTkButton(self.sidebar, text = "Entrenar Perceptron", state = "disabled", fg_color = "#8e44ad", hover_color = "#732d91")
        self.btn_perceptron.pack(pady = 10, padx = 20)


        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.canvas = tk.Canvas(self.main_frame, bg="#2b2b2b", bd=0, highlightthickness=1, highlightbackground="#3d3d3d")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


        self.panel_leyenda = ctk.CTkFrame(self.main_frame, width=200)
        self.panel_leyenda.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        self.panel_leyenda.pack_propagate(False) 
        ctk.CTkLabel(self.panel_leyenda, text="LEYENDA", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=15)
        self.leyenda_items = ctk.CTkFrame(self.panel_leyenda, fg_color="transparent")
        self.leyenda_items.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)


    def dibujar_imagen(self, img_tk, width, height):
        self.canvas.delete("all")
        self.canvas.create_image(self.offset, self.offset, image=img_tk, anchor=tk.NW)
        self.dibujar_ejes(width, height)

    def dibujar_ejes(self, w, h):
        c = "gray70"
        self.canvas.create_line(self.offset, self.offset-5, self.offset+w, self.offset-5, fill=c, width=2)
        paso_x = max(1, w // 4)
        for i in range(0, w + 1, paso_x):
            self.canvas.create_line(self.offset + i, self.offset - 5, self.offset + i, self.offset - 10, fill=c)
            self.canvas.create_text(self.offset + i, self.offset - 20, text=str(int(i)), font=("Helvetica", 9, "bold"), fill=c)

        self.canvas.create_line(self.offset-5, self.offset, self.offset-5, self.offset+h, fill=c, width=2)
        paso_y = max(1, h // 3)
        for i in range(0, h + 1, paso_y):
            self.canvas.create_line(self.offset - 5, self.offset + i, self.offset - 10, self.offset + i, fill=c)
            self.canvas.create_text(self.offset - 30, self.offset + i, text=str(int(i)), font=("Helvetica", 9, "bold"), fill=c)

    def dibujar_punto(self, x, y, color):
        self.canvas.create_oval(x-2 + self.offset, y-2 + self.offset, x+2 + self.offset, y+2 + self.offset, fill=color, outline="black", tags="puntos")

    def dibujar_centroide(self, x, y):
        self.canvas.create_rectangle(x-5 + self.offset, y-5 + self.offset, x+5 + self.offset, y+5 + self.offset, fill="yellow", outline="black", tags="puntos")

    def dibujar_error(self, x, y):
        self.canvas.create_text(x + self.offset, y + self.offset, text="X", fill="black", font=("Helvetica", 11, "bold"), tags="marca_error")
        self.canvas.create_text(x + self.offset, y + self.offset, text="X", fill="red", font=("Helvetica", 9, "bold"), tags="marca_error")

    def actualizar_leyenda(self, nombres, colores):
        for w in self.leyenda_items.winfo_children(): w.destroy()
        for i in range(len(nombres)):
            row = ctk.CTkFrame(self.leyenda_items, fg_color="transparent")
            row.pack(side=tk.TOP, fill=tk.X, pady=5)
            tk.Frame(row, width=15, height=15, bg=colores[i], bd=1, relief="solid").pack(side=tk.LEFT, padx=(0, 10))
            ctk.CTkLabel(row, text=f"Clase: {nombres[i]}", font=ctk.CTkFont(size=12)).pack(side=tk.LEFT)

    def mostrar_alerta(self, titulo, mensaje):
        messagebox.showinfo(titulo, mensaje)

    def mostrar_error(self, titulo, mensaje):
        messagebox.showerror(titulo, mensaje)

    def dibujar_linea_division(self, w_x, w_y, bias, w_img, h_img):
        
        self.canvas.delete("linea_division")
        
        
        if w_y == 0: 
            return 
            
        
        y0 = -bias / w_y
        
        
        y1 = -(w_x * w_img + bias) / w_y
        
        
        px1, py1 = 0 + self.offset, y0 + self.offset
        px2, py2 = w_img + self.offset, y1 + self.offset
        
        
        self.canvas.create_line(px1, py1, px2, py2, fill="#00FF00", width=4, tags="linea_division")

    def pedir_directorio(self):
        return ctk.filedialog.askdirectory(title="Selecciona el Dataset")
    
    def mostrar_resultado_kmeans(self, img_pil_segmentada):
        ventana_res = ctk.CTkToplevel(self)
        ventana_res.title("Resultado de Validación K-Means")
        # Hacemos la ventana un poco más grande que la imagen
        ventana_res.geometry(f"{img_pil_segmentada.width + 40}x{img_pil_segmentada.height + 40}")
        ventana_res.grab_set()
        
        # Convertimos la imagen para que se vea nítida
        img_tk = ctk.CTkImage(light_image=img_pil_segmentada, dark_image=img_pil_segmentada, size=(img_pil_segmentada.width, img_pil_segmentada.height))
        
        lbl_img = ctk.CTkLabel(ventana_res, image=img_tk, text="")
        lbl_img.pack(pady=20)