# Beltran_Saucedo_Axel_Alejandro_Practica5_Dep1.py
from modelo import ClasificadorModelo
from vista import VistaPrincipal
from docx import Document
from docx.shared import Inches, Pt
import matplotlib.pyplot as plt
import io
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

class Presentador:
    def __init__(self, vista: VistaPrincipal, modelo: ClasificadorModelo):
        self.vista = vista
        self.modelo = modelo
        
        # Variables de estado del Presentador
        self.img_pil = None
        self.img_tk = None # Guardamos referencia para que el Canvas no la borre
        self.img_width = 0
        self.img_height = 0
        self.franjas_elegidas = []
        self.nombres_clases = []
        self.colores_clases = []

        self._conectar_eventos()

    def _conectar_eventos(self):
        # El Presentador "inyecta" las funciones en los botones de la Vista "tonta"
        self.vista.btn_subir.configure(command=self.accion_subir_imagen)
        self.vista.btn_franjas.configure(command=self.accion_elegir_franjas)
        self.vista.btn_generar.configure(command=self.accion_generar_puntos)
        self.vista.btn_evaluar.configure(command=self.accion_evaluar)

    def accion_subir_imagen(self):
        filename = filedialog.askopenfilename(title="Selecciona imagen", filetypes=(("Imágenes", "*.png *.jpg *.jpeg"), ("Todos", "*.*")))
        if not filename: return
        
        try:
            self.img_pil = Image.open(filename).convert('RGB')
            self.img_width, self.img_height = self.img_pil.size
            
            # --- NUEVA LÓGICA DE ESCALADO ---
            # Forzamos a que la imagen ocupe 850 pixeles de ancho y calculamos su altura
            target_w = 850
            ratio = target_w / self.img_width
            self.img_width = target_w
            self.img_height = int(self.img_height * ratio)
            
            # Aplicamos el redimensionamiento con alta calidad (LANCZOS)
            self.img_pil = self.img_pil.resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)

            self.img_tk = ImageTk.PhotoImage(self.img_pil)
            self.vista.dibujar_imagen(self.img_tk, self.img_width, self.img_height)
            self.vista.btn_franjas.configure(state="normal")
        except Exception as e:
            self.vista.mostrar_error("Error", str(e))

    def accion_elegir_franjas(self):
        # Creamos el popup de configuración desde el Presentador
        popup = ctk.CTkToplevel(self.vista)
        popup.title('Configurar Clases')
        popup.geometry("350x420")
        popup.grab_set()

        ctk.CTkLabel(popup, text='¿Cuántas clases quieres?', font=ctk.CTkFont(weight="bold")).pack(pady=(15, 5))
        num_clases_var = tk.IntVar(value=2)
        ctk.CTkRadioButton(popup, text='2 clases', variable=num_clases_var, value=2).pack(pady=5)
        ctk.CTkRadioButton(popup, text='3 clases', variable=num_clases_var, value=3).pack(pady=5)

        opciones = ['1. Blanco', '2. Azul', '3. Rojo']
        var_c1, var_c2, var_c3 = tk.StringVar(value=opciones[0]), tk.StringVar(value=opciones[1]), tk.StringVar(value=opciones[2])

        def crear_fila(parent, texto, var):
            f = ctk.CTkFrame(parent, fg_color="transparent")
            f.pack(pady=5)
            ctk.CTkLabel(f, text=texto, width=60).pack(side=tk.LEFT, padx=10)
            ctk.CTkOptionMenu(f, variable=var, values=opciones).pack(side=tk.LEFT)
            return f

        crear_fila(popup, "Clase 1:", var_c1)
        crear_fila(popup, "Clase 2:", var_c2)
        
        f3 = ctk.CTkFrame(popup, fg_color="transparent")
        f3.pack(pady=5)
        lbl_c3 = ctk.CTkLabel(f3, text="Clase 3:", width=60).pack(side=tk.LEFT, padx=10)
        opt_c3 = ctk.CTkOptionMenu(f3, variable=var_c3, values=opciones)
        opt_c3.pack(side=tk.LEFT)

        def toggle_c3(*args):
            opt_c3.configure(state="disabled" if num_clases_var.get() == 2 else "normal")
        num_clases_var.trace_add('write', toggle_c3)
        toggle_c3()

        def guardar():
            seleccionados = [opciones.index(var_c1.get()), opciones.index(var_c2.get())]
            if num_clases_var.get() == 3: seleccionados.append(opciones.index(var_c3.get()))

            if len(set(seleccionados)) < len(seleccionados):
                self.vista.mostrar_error("Error", "No puedes repetir franjas.")
                return
            
            self.franjas_elegidas = seleccionados
            popup.destroy()
            self.vista.btn_generar.configure(state="normal")

        ctk.CTkButton(popup, text="Guardar Configuración", command=guardar).pack(pady=30)

    def accion_generar_puntos(self):
        try: num_ele = int(self.vista.entry_elementos.get())
        except: return self.vista.mostrar_error("Error", "Ingresa un número entero.")

        self.vista.canvas.delete('puntos')
        self.vista.canvas.delete('marca_error')
        self.nombres_clases, self.colores_clases = [], []

        h = self.img_height
        limites_y = [(0, h/3), (h/3, 2*h/3), (2*h/3, h)]
        nom_base, col_base = ['Blanco', 'Azul', 'Rojo'], ['#e0e0e0', 'blue', 'red']
        
        listas_rgb, listas_xy = [], []

        for idx in self.franjas_elegidas:
            y_min, y_max = limites_y[idx]
            
            # El eje X sigue siendo aleatorio en todo el ancho de la imagen
            eje_x = np.random.randint(0, self.img_width, num_ele)
            
            # --- NUEVA LÓGICA DE DISPERSIÓN (EJE Y) ---
            centro_y = (y_min + y_max) / 2
            
            # Este multiplicador (0.4) es tu "Factor de Ruido". 
            # Si lo subes a 0.6 o 0.8, los puntos invadirán más las otras franjas.
            dispersion = (y_max - y_min) * 0.7 
            
            # Generamos distribución Gaussiana (Campana de Gauss)
            eje_y_gauss = np.random.normal(loc=centro_y, scale=dispersion, size=num_ele)
            
            # Recortamos los puntos que se salgan por arriba (0) o por abajo (img_height-1)
            eje_y_gauss = np.clip(eje_y_gauss, 0, self.img_height - 1)
            eje_y = eje_y_gauss.astype(int)
            # ------------------------------------------
            
            R, G, B = [], [], []
            for x, y in zip(eje_x, eje_y):
                 r,g,b = self.img_pil.getpixel((x,y))
                 R.append(r); G.append(g); B.append(b)
            # ... (el resto del ciclo se queda exactamente igual) ...

            listas_rgb.append(np.array([R,G,B]))
            listas_xy.append(np.array([eje_x, eje_y]))
            self.nombres_clases.append(nom_base[idx])
            self.colores_clases.append(col_base[idx])

            # Le decimos a la Vista que dibuje
            for i in range(num_ele):
                self.vista.dibujar_punto(eje_x[i], eje_y[i], col_base[idx])
            self.vista.dibujar_centroide(np.mean(eje_x), np.mean(eje_y))

        # Le pasamos los datos al Modelo
        self.modelo.cargar_datos(listas_rgb, listas_xy, num_ele)
        
        self.vista.actualizar_leyenda(self.nombres_clases, self.colores_clases)
        self.vista.btn_evaluar.configure(state="normal")
        self.vista.mostrar_alerta('Éxito', "Puntos generados y modelo entrenado.")

    def accion_evaluar(self):
        popup = ctk.CTkToplevel(self.vista)
        popup.title('Evaluación')
        popup.geometry('300x250')
        popup.grab_set()

        ctk.CTkLabel(popup, text='Selecciona la Distancia:', font=ctk.CTkFont(weight="bold")).pack(pady=(15,10))
        var_dist = tk.IntVar(value=1)
        ctk.CTkRadioButton(popup, text='Euclidiana', variable=var_dist, value=1).pack(pady=5)
        ctk.CTkRadioButton(popup, text='Mahalanobis', variable=var_dist, value=2).pack(pady=5)
        ctk.CTkRadioButton(popup, text='Bayes', variable=var_dist, value=3).pack(pady=5)

        def procesar():
            dist = var_dist.get()
            popup.destroy()
            nom_dist = {1: 'Euclidiana', 2: 'Mahalanobis', 3: 'Bayes'}
            
            # 1. Ejecutamos Resustitución y Leave-One-Out
            # Recibimos las 4 variables de cada uno
            e_res, err_res, mc_res, efi_c_res = self.modelo.evaluar_resustitucion(dist)
            e_loo, err_loo, mc_loo, efi_c_loo = self.modelo.evaluar_leave_one_out(dist)
            
            # --- 2. EL MARATÓN CROSS-VALIDATION (50/50, 20 VUELTAS) ---
            # ¡NUEVO! Atrapamos también la lista de las 20 matrices individuales
            efi_global_avg_cro, efi_c_avg_cro, total_errores_xy_cro, vueltas_efi_cro, todas_las_matrices_cv = self.ejecutar_maraton_cross_validation(dist)
            
            # 3. ACTUALIZAMOS LA VISTA GRÁFICA (esto se queda igual)
            # Dibujamos en el Canvas los errores de Leave-One-Out
            self.vista.canvas.delete('marca_error')
            for (x, y) in err_loo: self.vista.dibujar_error(x, y)
                
            # Dibujamos la gráfica multilínea comparativa
            self._dibujar_grafica_lineas(efi_c_res, efi_c_avg_cro, efi_c_loo, e_res, efi_global_avg_cro, e_loo, dist)

            # --- 4. LA NUEVA LÓGICA DE DOCUMENTOS DOCX ---
            # Aviso de "pensando" porque esto se va a tardar un poco
            popup_wait = ctk.CTkToplevel(self.vista)
            popup_wait.geometry("300x100")
            popup_wait.title("Pensando...")
            ctk.CTkLabel(popup_wait, text="Generando los 4 documentos Word.\nPor favor espera...").pack(pady=20)
            self.vista.update()

            # Forzamos la misma escala para todas las gráficas de clases (0-100%)
            y_ticks = np.arange(0, 110, 10)

            # --- DOCUMENTO 1: RESUSTITUCIÓN ---
            doc_res = Document()
            self._generar_documento_docx(doc_res, "Resustitucion", nom_dist[dist], mc_res, efi_c_res, e_res, y_ticks)
            doc_res.save(f"Beltran_Saucedo_Axel_Reporte_Resustitucion_{nom_dist[dist]}.docx")

            # --- DOCUMENTO 2: CROSS-VALIDATION (50/50) ---
            doc_cro = Document()
            # ¡NUEVO! Esta función recibe la lista con las 20 matrices
            self._generar_documento_cv_20_vueltas_docx(doc_cro, nom_dist[dist], todas_las_matrices_cv, vueltas_efi_cro, efi_c_avg_cro, efi_global_avg_cro, y_ticks)
            doc_cro.save(f"Beltran_Saucedo_Axel_Reporte_Cross_Validation_{nom_dist[dist]}.docx")

            # --- DOCUMENTO 3: LEAVE-ONE-OUT ---
            doc_loo = Document()
            self._generar_documento_docx(doc_loo, "Leave_One_Out", nom_dist[dist], mc_loo, efi_c_loo, e_loo, y_ticks)
            doc_loo.save(f"Beltran_Saucedo_Axel_Reporte_Leave_One_Out_{nom_dist[dist]}.docx")

            # --- DOCUMENTO 4: COMPARATIVO FINAL ---
            doc_comp = Document()
            self._generar_documento_comparativo_final_docx(doc_comp, nom_dist[dist], efi_c_res, efi_c_avg_cro, efi_c_loo, e_res, efi_global_avg_cro, e_loo)
            doc_comp.save(f"Beltran_Saucedo_Axel_Reporte_Comparativo_3_Metodos_{nom_dist[dist]}.docx")

            popup_wait.destroy()
            self.vista.mostrar_alerta("¡Éxito!", "Se han generado 4 documentos .docx en la carpeta de tu proyecto.")

        ctk.CTkButton(popup, text="Iniciar Comparativa", command=procesar).pack(pady=20)

    def _dibujar_grafica_lineas(self, e_c_res, e_c_cro, e_c_loo, g_res, g_cro, g_loo, dist):
        vent = ctk.CTkToplevel(self.vista)
        nom = {1: 'Euclidiana', 2: 'Mahalanobis', 3: 'Bayes'}
        vent.title(f"Gráfica Comparativa Multilínea - {nom[dist]}")
        vent.geometry("600x450")
        vent.grab_set()

        ctk.CTkLabel(vent, text=f"Eficiencia por Clase y Método ({nom[dist]})", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)


        dic_resultados = {'Resustitución': g_res, 'Cross-Validation': g_cro, 'Leave-One-Out': g_loo}
        metodo_ganador =  max(dic_resultados, key=dic_resultados.get)
        val_ganador = dic_resultados[metodo_ganador]

        ctk.CTkLabel(vent, text = f'El ganador es: {metodo_ganador} ({val_ganador:.1f}%)', font=ctk.CTkFont(size=14, weight="bold", slant="italic"), text_color="#FFD700").pack(pady=(0, 5))
        
        
        # Canvas más ancho para que quepa la leyenda a la derecha
        c = tk.Canvas(vent, width=560, height=300, bg="#2b2b2b", bd=0, highlightthickness=0)
        c.pack(pady=10)

        # Dimensiones del plano
        margen_izq = 50
        margen_inf = 250
        alto_graf = 200
        ancho_graf = 300

        # Ejes X y Y
        c.create_line(margen_izq, margen_inf, margen_izq + ancho_graf + 20, margen_inf, width=2, fill="gray70") 
        c.create_line(margen_izq, margen_inf - alto_graf - 20, margen_izq, margen_inf, width=2, fill="gray70") 

        # Etiquetas del Eje Y
        c.create_text(margen_izq - 20, margen_inf - alto_graf, text="100%", font=("Helvetica", 9), fill="white")
        c.create_text(margen_izq - 20, margen_inf, text="0%", font=("Helvetica", 9), fill="white")

        num_clases = len(self.nombres_clases)
        paso_x = ancho_graf / (num_clases - 1) if num_clases > 1 else ancho_graf

        # Dibujar etiquetas del Eje X (Nombres de tus clases)
        coordenadas_x = []
        for i, nombre in enumerate(self.nombres_clases):
            x = margen_izq + (i * paso_x)
            coordenadas_x.append(x)
            c.create_text(x, margen_inf + 15, text=nombre, font=("Helvetica", 10, "bold"), fill="white")

        # --- FUNCIÓN INTERNA PARA TRAZAR CADA LÍNEA ---
        def trazar_linea(eficiencias, color, nombre_metodo, efi_global, offset_leyenda_y):
            puntos = []
            for i, efi in enumerate(eficiencias):
                x = coordenadas_x[i]
                y = margen_inf - (efi / 100.0 * alto_graf)
                puntos.append((x, y))
                # Dibujamos el "puntito"
                c.create_oval(x-4, y-4, x+4, y+4, fill=color, outline="white")
                # Ponemos el % exacto arribita del punto
                c.create_text(x, y-12, text=f"{efi:.1f}", font=("Helvetica", 8), fill=color)

            # Trazamos la línea conectando los puntos
            for i in range(len(puntos)-1):
                c.create_line(puntos[i][0], puntos[i][1], puntos[i+1][0], puntos[i+1][1], fill=color, width=2, dash=(4,2))

            # Dibujamos la leyenda al estilo del profe (Global -> Método)
            lx = margen_izq + ancho_graf + 40
            ly = margen_inf - alto_graf + offset_leyenda_y
            c.create_line(lx, ly, lx+30, ly, fill=color, width=2, dash=(4,2))
            c.create_text(lx+35, ly, text=f"{efi_global:.1f}% -> {nombre_metodo}", font=("Helvetica", 11, "bold"), fill="white", anchor=tk.W)

        # Trazamos las 3 líneas con sus colores representativos
        trazar_linea(e_c_res, "#4CAF50", "Resustitución", g_res, 40)   # Línea Verde
        trazar_linea(e_c_loo, "#F44336", "Leave-One-Out", g_loo, 90)   # Línea Roja
        trazar_linea(e_c_cro, "#2196F3", "Cross-Validation", g_cro, 140) # Línea Azul

    # =========================================================================
    # NUEVAS FUNCIONES PARA EL MARATÓN DE CROSS-VALIDATION (MVP - MVP)
    # =========================================================================
    def ejecutar_maraton_cross_validation(self, dist):
        num_vueltas = 20
        vueltas_efi_global = []
        vueltas_efi_por_clase = [] 
        todas_las_coordenadas_error = []
        
        # 1. Nueva lista para guardar las 20 matrices
        todas_las_matrices_cv = []

        for i in range(num_vueltas):
            efi_global_v, err_v, mc_v, efi_clases_v = self.modelo.evaluar_cross_validation(dist)
            vueltas_efi_global.append(efi_global_v)
            vueltas_efi_por_clase.append(efi_clases_v)
            todas_las_coordenadas_error.extend(err_v)
            
            # 2. Guardamos la matriz de esta vuelta
            todas_las_matrices_cv.append(mc_v)

        efi_global_promedio = np.mean(vueltas_efi_global)
        matriz_efi_por_clase = np.array(vueltas_efi_por_clase)
        efi_por_clase_promedio = np.mean(matriz_efi_por_clase, axis=0)

        # Devolvemos 5 cosas
        return efi_global_promedio, efi_por_clase_promedio, todas_las_coordenadas_error, vueltas_efi_global, todas_las_matrices_cv

    def _generar_reporte_txt_cv_20_vueltas(self, nom_dist, vueltas_efi, efi_c_avg, efi_global_avg):
        """Genera un reporte especial con los 20 resultados individuales y el promedio."""
        filename = f"Reporte_Cross_Validation_Detallado_20_Vueltas.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"=== REPORTE DETALLADO DE CROSS-VALIDATION (20 VUELTAS) ===\n")
            f.write(f"Metodo de Evaluacion: Repeated Random Sub-sampling Validation\n")
            f.write(f"Distancia Matematica: {nom_dist}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("--- EFICIENCIA GLOBAL POR ITERACION ---\n")
            f.write(f"{'Vuelta':<8} | {'Eficiencia':<10}\n")
            f.write("-" * 22 + "\n")
            for i, efi in enumerate(vueltas_efi):
                f.write(f"{i+1:^{8}} | {efi:^{10}.2f}%\n")
            
            f.write("\n" + "=" * 31 + "\n")
            f.write(f"--- PROMEDIO FINAL (Normalizado) ---\n")
            f.write(f"EFICIENCIA GLOBAL PROMEDIO: {efi_global_avg:.2f}%\n")
            f.write("=" * 31 + "\n\n")

            f.write("--- PROMEDIO DE EFICIENCIA POR CLASE ---\n")
            for i, efi in enumerate(efi_c_avg):
                f.write(f"{self.nombres_clases[i]}: {efi:.2f}%\n")
    
    # =========================================================================
    # NUEVAS FUNCIONES DE GENERACIÓN DE DOCUMENTOS DOCX (MVP - MVP)
    # =========================================================================
    def _generar_documento_docx(self, doc, metodo_eval, nom_dist, matriz, efi_clases, efi_global, y_ticks):
        # 1. Títulos básicos
        doc.add_heading(f'Reporte de Clasificador RGB - Práctica 5', 0)
        doc.add_paragraph(f'Metodo de Evaluación: {metodo_eval}').bold = True
        doc.add_paragraph(f'Distancia Matematica: {nom_dist}')
        doc.add_paragraph(f'Eficiencia Global: {efi_global:.2f}%').bold = True
        doc.add_paragraph("-" * 60)

        # 2. Matriz de Confusión como tabla dinámica en Word
        doc.add_heading('--- MATRIZ DE CONFUSIÓN ---', level=2)
        rows = len(matriz) + 1
        cols = len(matriz[0]) + 1
        table = doc.add_table(rows=rows, cols=cols)
        table.style = 'Table Grid'
        
        # Encabezado (Clasificado)
        table.cell(0, 0).text = "Real \\ Clasif"
        for i, nombre in enumerate(self.nombres_clases):
            table.cell(0, i+1).text = nombre
        
        # Filas y datos
        for i, nombre in enumerate(self.nombres_clases):
            table.cell(i+1, 0).text = nombre
            for j, val in enumerate(matriz[i]):
                table.cell(i+1, j+1).text = str(val)

        # 3. Eficiencia por Clase
        doc.add_heading('--- EFICIENCIA POR CLASE ---', level=2)
        for i, nombre in enumerate(self.nombres_clases):
            doc.add_paragraph(f"{nombre}: {efi_clases[i]:.2f}%")

        # 4. Gráfica de Clases integrada (Matplotlib -> BytesIO -> Word)
        self._agregar_grafica_clases_a_docx(doc, efi_clases, y_ticks)

    def _generar_documento_cv_20_vueltas_docx(self, doc, nom_dist, lista_matrices, vueltas_efi, efi_c_avg, efi_global_avg, y_ticks):
        # 1. Títulos básicos
        doc.add_heading(f'Reporte de Cross-Validation (20 Vueltas) - Práctica 5', 0)
        doc.add_paragraph(f'Distancia Matematica: {nom_dist}')
        doc.add_paragraph(f'Eficiencia Global Promedio (Normalizada): {efi_global_avg:.2f}%').bold = True
        doc.add_paragraph("-" * 60)

        # 2. Eficiencia Global por Iteración
        doc.add_heading('--- EFICIENCIA GLOBAL POR ITERACIÓN ---', level=2)
        table_efi = doc.add_table(rows=21, cols=2)
        table_efi.style = 'Table Grid'
        table_efi.cell(0, 0).text = "Vuelta"
        table_efi.cell(0, 1).text = "Eficiencia %"
        for i, efi in enumerate(vueltas_efi):
            table_efi.cell(i+1, 0).text = str(i+1)
            table_efi.cell(i+1, 1).text = f"{efi:.2f}%"

        # 3. LAS 20 MATRICES DE CONFUSIÓN
        doc.add_page_break() # Nueva página
        doc.add_heading('--- LAS 20 MATRICES DE CONFUSIÓN (Detalle) ---', level=1)
        for i, mc in enumerate(lista_matrices):
            doc.add_paragraph(f"Matriz de Confusión # {i+1} (Vuelta {i+1}):", style='Subtitle')
            self._agregar_matriz_como_tabla_docx(doc, mc)
            doc.add_paragraph("") # Espacio

        # 4. Promedio de Eficiencia por Clase
        doc.add_heading('--- PROMEDIO DE EFICIENCIA POR CLASE ---', level=2)
        for i, nombre in enumerate(self.nombres_clases):
            doc.add_paragraph(f"{nombre}: {efi_c_avg[i]:.2f}%")

        # 5. Gráfica de Clases integrad
        self._agregar_grafica_clases_a_docx(doc, efi_c_avg, y_ticks)

    def _generar_documento_comparativo_final_docx(self, doc, nom_dist, e_c_res, e_c_cro, e_c_loo, g_res, g_cro, g_loo):
        # 1. Títulos básicos
        doc.add_heading(f'Reporte Comparativo de 3 Métodos - Práctica 5', 0)
        doc.add_paragraph(f'Distancia Matematica: {nom_dist}')
        doc.add_paragraph("-" * 60)

        # 2. Resumen de Eficiencias Globales
        doc.add_heading('--- RESUMEN DE EFICIENCIAS GLOBALES ---', level=2)
        doc.add_paragraph(f"1. Resustitucion: {g_res:.2f}%")
        doc.add_paragraph(f"2. Cross-Validation (Promedio 20 Vueltas): {g_cro:.2f}%")
        doc.add_paragraph(f"3. Leave-One-Out (50/50): {g_loo:.2f}%")

        # 3. La Gráfica Comparativa (Estilo de barras formal)
        doc.add_heading('--- GRÁFICA COMPARATIVA DE EFICIENCIA GLOBAL ---', level=2)
        
        # Generamos la gráfica con Matplotlib (Fondo blanco para documento de Word)
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        metodos = ['Resustitucion', 'Cross Validation', 'Leave One Out']
        valores = [g_res, g_cro, g_loo]
        colores = ['#00639a', '#6a0dad', '#006d2c'] # Azul, Morado, Verde oscuro (como la foto)
        
        # Dibujamos las barras
        barras = ax.bar(metodos, valores, color=colores, width=0.6)
        
        # Ajustamos el Eje Y de 0 a 115 para dar espacio al texto arriba
        ax.set_ylim(0, 115)
        y_ticks = np.arange(0, 120, 20)
        ax.set_yticks(y_ticks)
        ax.set_ylabel("Eficiencia Global (Accuracy %)", fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7) # Cuadrícula punteada suave
        
        ax.set_title(f"Comparativa de Métodos de Validación\nDistancia: {nom_dist}", fontweight='bold', pad=15)
        
        # Etiquetas de porcentaje sobre cada barra
        for barra in barras:
            altura = barra.get_height()
            ax.text(barra.get_x() + barra.get_width()/2., altura + 2,
                    f'{altura:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Pasamos la gráfica a Word con alta resolución (dpi=150)
        memfile = io.BytesIO()
        fig.savefig(memfile, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig) # Cerramos la figura
        
        doc.add_picture(memfile, width=Inches(6.0))
        
    # --- FUNCIONES AUXILIARES PARA GRAFICACIÓN Y WORD ---
    def _agregar_matriz_como_tabla_docx(self, doc, matriz):
        """Agrega una matriz de confusión individual como una tabla Grid simple en Word."""
        num_clases = len(self.nombres_clases)
        table = doc.add_table(rows=num_clases, cols=num_clases)
        table.style = 'Table Grid'
        
        for i in range(num_clases):
            for j in range(num_clases):
                table.cell(i, j).text = str(matriz[i][j])

    def _agregar_grafica_clases_a_docx(self, doc, efi_clases, y_ticks):
        """Genera una gráfica de Matplotlib para la eficiencia por clase y la integra a Word."""
        fig, ax = plt.subplots(figsize=(6, 3))
        num_clases = len(self.nombres_clases)
        x = range(num_clases)
        
        # Color dorado para el Canvas de Clases, color del punto para la barra
        ax.bar(x, efi_clases, color=self.colores_clases, width=0.6, align='center')
        
        # Forzamos la misma escala (0-100%)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{tick}%" for tick in y_ticks])
        ax.grid(axis='y', linestyle='--')
        
        ax.set_xticks(x)
        ax.set_xticklabels(self.nombres_clases, fontweight='bold')
        ax.set_ylabel("Eficiencia", fontweight='bold')
        
        # Etiquetamos el valor encima de la barra
        for i, val in enumerate(efi_clases):
            ax.text(i, val+2, f"{val:.1f}%", ha='center', fontsize=9, fontweight='bold')

        # Pasamos la gráfica a Word
        memfile = io.BytesIO()
        fig.savefig(memfile, format='png', bbox_inches='tight')
        plt.close(fig) # Cerramos la figura
        
        doc.add_paragraph("Gráfica de barras - Eficiencia por Clase:")
        doc.add_picture(memfile, width=Inches(5.5))       

if __name__ == "__main__":
    app_vista = VistaPrincipal()
    app_modelo = ClasificadorModelo()
    
    # Arrancamos el director de orquesta
    presentador = Presentador(app_vista, app_modelo)
    
    app_vista.mainloop()