# modelo.py
import numpy as np

class ClasificadorModelo:
    def __init__(self):
        self.clases_rgb = []
        self.clases_xy = []
        self.medias = []
        self.covarianzas = []
        self.num_clases = 0
        self.num_ele = 0

    def cargar_datos(self, lista_rgb, lista_xy, num_ele):
        self.clases_rgb = lista_rgb
        self.clases_xy = lista_xy
        self.num_clases = len(lista_rgb)
        self.num_ele = num_ele

        self.medias = []
        self.covarianzas = []
        
        for matriz_rgb in self.clases_rgb:
            media = np.mean(matriz_rgb, axis=1)
            cov = np.cov(matriz_rgb) + np.eye(3) * 1e-5 # Regularización
            self.medias.append(media)
            self.covarianzas.append(cov)

    def evaluar_resustitucion(self, opcion_dist):
        aciertos = 0
        total_eva = self.num_ele * self.num_clases
        errores_xy = [] 
        
        # 1. Creamos la Matriz de Confusión vacía (ej. 3x3)
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
                
                # 2. Registramos el "voto" en la Matriz de Confusión
                matriz_confusion[clase_real][clase_ganadora] += 1
                
                if clase_ganadora == clase_real: aciertos += 1
                else: errores_xy.append((punto_x, punto_y)) 

        # 3. Cálculos finales
        eficiencia_global = (aciertos / total_eva) * 100
        
        # La diagonal tiene los aciertos. Dividimos entre el total de esa fila para sacar el % por clase.
        eficiencia_por_clase = (np.diag(matriz_confusion) / np.sum(matriz_confusion, axis=1)) * 100

        # Ahora devolvemos 4 cosas en lugar de 2
        return eficiencia_global, errores_xy, matriz_confusion, eficiencia_por_clase

    def evaluar_cross_validation(self, opcion_dist):
        aciertos = 0
        mitad = self.num_ele // 2
        total_evaluados = (self.num_ele - mitad) * self.num_clases
        errores_xy = []
        matriz_confusion = np.zeros((self.num_clases, self.num_clases), dtype=int)

        medias_ent, cov_ent, inv_covs, det_covs = [], [], [], []

        # --- EL SECRETO DEL MARATÓN: BARAJEAR LOS DATOS ---
        # Creamos una lista de índices aleatorios para que la mitad siempre sea distinta
        indices_aleatorios = np.random.permutation(self.num_ele)
        indices_entrenamiento = indices_aleatorios[:mitad]
        indices_prueba = indices_aleatorios[mitad:]
        # --------------------------------------------------

        # 1. Fase de entrenamiento (usando la mitad aleatoria)
        for clase_real in range(self.num_clases):
            puntos_ent = self.clases_rgb[clase_real][:, indices_entrenamiento]
            medias_ent.append(np.mean(puntos_ent, axis=1))
            cov_ent.append(np.cov(puntos_ent) + np.eye(3) * 1e-5)

        if opcion_dist in [2, 3]:
            for cov in cov_ent:
                inv_covs.append(np.linalg.inv(cov))
                det_covs.append(np.linalg.det(cov))

        # 2. Fase de prueba (usando la otra mitad aleatoria)
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
        
        # Matriz de confusión vacía
        matriz_confusion = np.zeros((self.num_clases, self.num_clases), dtype=int)
        
        for clase_real in range(self.num_clases):
            for i in range(self.num_ele):
                vector = self.clases_rgb[clase_real][:, i]
                punto_x, punto_y = self.clases_xy[clase_real][0, i], self.clases_xy[clase_real][1, i]
                
                medias_loo, inv_covs, det_covs = [], [], []
                
                for clase_ent in range(self.num_clases):
                    puntos_ent = np.delete(self.clases_rgb[clase_ent], i, axis=1) if clase_ent == clase_real else self.clases_rgb[clase_ent]
                    cov_loc = np.cov(puntos_ent) + np.eye(3) * 1e-5
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
                
                # Sumamos el voto a la matriz
                matriz_confusion[clase_real][clase_ganadora] += 1
                
                if clase_ganadora == clase_real: aciertos += 1
                else: errores_xy.append((punto_x, punto_y))
                    
        eficiencia_global = (aciertos / total_evaluados) * 100
        suma_filas = np.sum(matriz_confusion, axis=1)
        suma_filas[suma_filas == 0] = 1 
        eficiencia_por_clase = (np.diag(matriz_confusion) / suma_filas) * 100
        
        # ¡AQUÍ ESTÁ LA CLAVE! Devolvemos 4 cosas:
        return eficiencia_global, errores_xy, matriz_confusion, eficiencia_por_clase