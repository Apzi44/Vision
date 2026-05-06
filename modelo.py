import numpy as np
import os
from PIL import Image
from sklearn.cluster import KMeans

class ClasificadorModelo:
    def __init__(self):
        self.clases_rgb = []
        self.clases_xy = []
        self.medias = []
        self.covarianzas = []
        self.num_clases = 0
        self.num_ele = 0

    def cargar_dataset_masivo(self, ruta_carpeta, num_descriptores=1000):
        archivos_validos = [".jpg", ".jpeg", ".png"]
        mega_dataset = []
        imagenes_leidas = 0
        
        # 1. Explorar todos los archivos en la carpeta que le pasemos
        for archivo in os.listdir(ruta_carpeta):
            ext = os.path.splitext(archivo)[1].lower()
            
            if ext in archivos_validos:
                ruta_completa = os.path.join(ruta_carpeta, archivo)
                
                try:
                    # 2. Abrir la imagen y asegurar que esté en formato RGB
                    img = Image.open(ruta_completa).convert('RGB')
                    img_np = np.array(img)
                    
                    # 3. Aplanar la imagen: pasar de 3D (Alto, Ancho, RGB) a una lista 2D (Total Pixeles, RGB)
                    pixeles_planos = img_np.reshape(-1, 3)
                    total_pixeles = pixeles_planos.shape[0]
                    
                    # 4. Elegir puntos al azar (Si la imagen es muy chiquita, tomamos lo que se pueda)
                    puntos_a_extraer = min(num_descriptores, total_pixeles)
                    
                    # Generamos índices aleatorios sin repetición
                    indices_aleatorios = np.random.choice(total_pixeles, puntos_a_extraer, replace=False)
                    
                    # 5. Extraemos solo los píxeles (descriptores) de esos índices
                    descriptores = pixeles_planos[indices_aleatorios]
                    
                    # 6. Los guardamos en la lista gigante
                    mega_dataset.append(descriptores)
                    imagenes_leidas += 1
                    
                except Exception as e:
                    print(f"Error al leer la imagen {archivo}: {e}")
                    
        # 7. Unir todas las mini-listas en un solo mega arreglo de Numpy
        if len(mega_dataset) > 0:
            # vstack apila los arreglos uno encima de otro
            dataset_final = np.vstack(mega_dataset)
            return True, dataset_final, imagenes_leidas
        else:
            return False, "No se encontraron imágenes válidas en la carpeta.", 0

    def entrenar_kmeans(self, dataset, k):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(dataset)
            self.modelo_kmeans = kmeans
            self.centroides = kmeans.cluster_centers_

            return True, self.centroides
        except Exception as e:
            return False, f"Error de entrenamiento: {e}"

    def cargar_datos(self, lista_rgb, lista_xy, num_ele):
        self.clases_rgb = lista_rgb
        self.clases_xy = lista_xy
        self.num_clases = len(lista_rgb)
        self.num_ele = num_ele

        self.medias = []
        self.covarianzas = []
        
        for matriz_rgb in self.clases_rgb:
            media = np.mean(matriz_rgb, axis=1)

            cov = np.cov(matriz_rgb, ddof=0) + np.eye(3) * 1e-5 
            self.medias.append(media)
            self.covarianzas.append(cov)

    def evaluar_resustitucion(self, opcion_dist):
        aciertos = 0
        total_eva = self.num_ele * self.num_clases
        errores_xy = [] 
        
        matriz_confusion = np.zeros((self.num_clases, self.num_clases), dtype=int)

        inv_covs, det_covs = [], []
        if opcion_dist in [2, 3]:
            for cov in self.covarianzas:
                inv_covs.append(np.linalg.inv(cov))
                det_covs.append(np.linalg.det(cov))

        for clase_real in range(self.num_clases):
            for i in range(self.num_ele):
                vector = self.clases_rgb[clase_real][:, i]
                punto_x = self.clases_xy[clase_real][0, i]
                punto_y = self.clases_xy[clase_real][1, i]
                
                dist_tot = np.zeros(self.num_clases)
                for j in range(self.num_clases):
                    dif = vector - self.medias[j]
                    if opcion_dist == 1: dist_tot[j] = np.linalg.norm(dif)
                    elif opcion_dist == 2: dist_tot[j] = np.sqrt(dif.T @ inv_covs[j] @ dif)
                    elif opcion_dist == 3:
                        cuad = dif.T @ inv_covs[j] @ dif
                        coef = 1 / (2 * np.pi * np.sqrt(det_covs[j]))
                        dist_tot[j] = (coef * np.exp(-0.5 * cuad)) * (1 / self.num_clases)

                clase_ganadora = np.argmax(dist_tot) if opcion_dist == 3 else np.argmin(dist_tot)
                
                matriz_confusion[clase_real][clase_ganadora] += 1
                
                if clase_ganadora == clase_real: aciertos += 1
                else: errores_xy.append((punto_x, punto_y)) 

        eficiencia_global = (aciertos / total_eva) * 100
        eficiencia_por_clase = (np.diag(matriz_confusion) / np.sum(matriz_confusion, axis=1)) * 100

        return eficiencia_global, errores_xy, matriz_confusion, eficiencia_por_clase

    def evaluar_cross_validation(self, opcion_dist):
        aciertos = 0
        mitad = self.num_ele // 2
        total_evaluados = (self.num_ele - mitad) * self.num_clases
        errores_xy = []
        matriz_confusion = np.zeros((self.num_clases, self.num_clases), dtype=int)

        medias_ent, cov_ent, inv_covs, det_covs = [], [], [], []

        indices_aleatorios = np.random.permutation(self.num_ele)
        indices_entrenamiento = indices_aleatorios[:mitad]
        indices_prueba = indices_aleatorios[mitad:]

        for clase_real in range(self.num_clases):
            puntos_ent = self.clases_rgb[clase_real][:, indices_entrenamiento]
            medias_ent.append(np.mean(puntos_ent, axis=1))

            cov_ent.append(np.cov(puntos_ent, ddof=0) + np.eye(3) * 1e-5)

        if opcion_dist in [2, 3]:
            for cov in cov_ent:
                inv_covs.append(np.linalg.inv(cov))
                det_covs.append(np.linalg.det(cov))

        for clase_real in range(self.num_clases):
            puntos_pru = self.clases_rgb[clase_real][:, indices_prueba]
            xy_pru = self.clases_xy[clase_real][:, indices_prueba]

            for i in range(puntos_pru.shape[1]):
                vector = puntos_pru[:, i]
                punto_x, punto_y = xy_pru[0, i], xy_pru[1, i]
                dist_tot = np.zeros(self.num_clases)

                for j in range(self.num_clases):
                    dif = vector - medias_ent[j]
                    if opcion_dist == 1: dist_tot[j] = np.linalg.norm(dif)
                    elif opcion_dist == 2: dist_tot[j] = np.sqrt(dif.T @ inv_covs[j] @ dif)
                    elif opcion_dist == 3:
                        cuad = dif.T @ inv_covs[j] @ dif
                        coef = 1 / (2 * np.pi * np.sqrt(det_covs[j]))
                        dist_tot[j] = (coef * np.exp(-0.5 * cuad)) * (1 / self.num_clases)

                clase_ganadora = np.argmax(dist_tot) if opcion_dist == 3 else np.argmin(dist_tot)
                
                matriz_confusion[clase_real][clase_ganadora] += 1
                
                if clase_ganadora == clase_real: aciertos += 1
                else: errores_xy.append((punto_x, punto_y))

        eficiencia_global = (aciertos / total_evaluados) * 100
        suma_filas = np.sum(matriz_confusion, axis=1)
        suma_filas[suma_filas == 0] = 1 
        eficiencia_por_clase = (np.diag(matriz_confusion) / suma_filas) * 100

        return eficiencia_global, errores_xy, matriz_confusion, eficiencia_por_clase

    def evaluar_leave_one_out(self, opcion_dist):
        aciertos = 0
        total_evaluados = self.num_ele * self.num_clases
        errores_xy = []
        matriz_confusion = np.zeros((self.num_clases, self.num_clases), dtype=int)
        
        for clase_real in range(self.num_clases):
            for i in range(self.num_ele):
                vector = self.clases_rgb[clase_real][:, i]
                punto_x, punto_y = self.clases_xy[clase_real][0, i], self.clases_xy[clase_real][1, i]
                
                medias_loo, inv_covs, det_covs = [], [], []
                
                for clase_ent in range(self.num_clases):
                    puntos_ent = np.delete(self.clases_rgb[clase_ent], i, axis=1) if clase_ent == clase_real else self.clases_rgb[clase_ent]

                    cov_loc = np.cov(puntos_ent, ddof=0) + np.eye(3) * 1e-5
                    medias_loo.append(np.mean(puntos_ent, axis=1))
                    
                    if opcion_dist in [2, 3]:
                        inv_covs.append(np.linalg.inv(cov_loc))
                        det_covs.append(np.linalg.det(cov_loc))
                        
                dist_tot = np.zeros(self.num_clases)
                for j in range(self.num_clases):
                    diff = vector - medias_loo[j]
                    if opcion_dist == 1: dist_tot[j] = np.linalg.norm(diff)
                    elif opcion_dist == 2: dist_tot[j] = np.sqrt(diff.T @ inv_covs[j] @ diff)
                    elif opcion_dist == 3:
                        cuad = diff.T @ inv_covs[j] @ diff
                        coef = 1 / (2 * np.pi * np.sqrt(det_covs[j]))
                        dist_tot[j] = (coef * np.exp(-0.5 * cuad)) * (1 / self.num_clases)
                        
                clase_ganadora = np.argmax(dist_tot) if opcion_dist == 3 else np.argmin(dist_tot)
                
                matriz_confusion[clase_real][clase_ganadora] += 1
                
                if clase_ganadora == clase_real: aciertos += 1
                else: errores_xy.append((punto_x, punto_y))
                    
        eficiencia_global = (aciertos / total_evaluados) * 100
        suma_filas = np.sum(matriz_confusion, axis=1)
        suma_filas[suma_filas == 0] = 1 
        eficiencia_por_clase = (np.diag(matriz_confusion) / suma_filas) * 100

    
    def entrenar_perceptron_xy(self, w_iniciales_xy, bias_inicial, r, max_epocas=5000):
        if self.num_clases != 2:
            return False, "El Perceptrón solo soporta 2 clases."
            
        c1_xy = self.clases_xy[0].T 
        c2_xy = self.clases_xy[1].T 
        
        pesos = np.array(w_iniciales_xy, dtype=float)
        bias = float(bias_inicial)
        tasa = float(r)
        
        X_datos = np.vstack((c1_xy, c2_xy))
        Y_esperada = np.concatenate((np.zeros(len(c1_xy)), np.ones(len(c2_xy))))        
        
        X_datos_norm = X_datos / 1000.0 
        
        for epoca in range(max_epocas):
            errores_epoca = 0
            
            
            for i in range(len(X_datos_norm)):
                pixel_coords = X_datos_norm[i] 
                clase_real = Y_esperada[i]
                
                f_sal = np.dot(pixel_coords, pesos) + bias
                
                if clase_real == 0 and f_sal >= 0:
                    pesos = pesos - (tasa * pixel_coords)
                    bias = bias - tasa
                    errores_epoca += 1
                elif clase_real == 1 and f_sal <= 0:
                    pesos = pesos + (tasa * pixel_coords)
                    bias = bias + tasa
                    errores_epoca += 1
                    
            if errores_epoca == 0:
                
                pesos_limpios = [round(float(w)/1000.0, 6) for w in pesos]
                bias_limpio = round(float(bias), 6)
                return True, (pesos_limpios, bias_limpio, epoca + 1)
                
        return False, "El Perceptrón no convergió. Intenta cuadros sin empalme." 
                
    def segmentar_con_kmeans(self, imagen_pil):
        # 1. Asegurarnos de que ya haya un modelo entrenado
        if getattr(self, 'modelo_kmeans', None) is None:
            return False, "Primero debes entrenar el algoritmo K-Means."
            
        try:
            # 2. Preparamos la imagen de prueba (Query Image)
            img_np = np.array(imagen_pil.convert('RGB'))
            alto, ancho, _ = img_np.shape
            
            # Aplanamos los píxeles
            pixeles_planos = img_np.reshape(-1, 3)
            
            # 3. LA PREDICCIÓN: K-Means clasifica cada píxel en su grupo más cercano
            etiquetas = self.modelo_kmeans.predict(pixeles_planos)
            
            # 4. Convertimos los centroides a colores RGB enteros (0-255)
            centroides_enteros = np.round(self.centroides).astype(np.uint8)
            
            # 5. Pintamos cada píxel con el color exacto de su centroide
            pixeles_segmentados = centroides_enteros[etiquetas]
            
            # 6. Reconstruimos el arreglo a la forma original de la foto (Alto x Ancho)
            imagen_final_np = pixeles_segmentados.reshape(alto, ancho, 3)
            
            return True, imagen_final_np
            
        except Exception as e:
            return False, f"Error en la segmentación: {e}"