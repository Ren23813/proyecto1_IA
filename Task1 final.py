#task 1.1 Discretización del mundo

import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from collections import deque

def procesar_laberinto(ruta_imagen, tile_size=10):
    # Cargar imagen y convertir a RGB
    img = Image.open(ruta_imagen).convert('RGB')
    ancho, alto = img.size
    
    # Calcular dimensiones de la matriz 
    columnas = ancho // tile_size
    filas = alto // tile_size
    
    grid = np.zeros((filas, columnas), dtype=int)
    inicio = None
    metas = []
    matriz_colores = np.zeros((filas, columnas, 3), dtype=np.uint8)

    for f in range(filas):
        for c in range(columnas):
            # Definir la caja del tile
            caja = (c * tile_size, f * tile_size, (c + 1) * tile_size, (f + 1) * tile_size)
            tile = img.crop(caja)
            
            color_medio = np.array(tile).mean(axis=(0, 1))
            r, g, b = color_medio
            matriz_colores[f, c] = [r, g, b]
            
            if r < 50 and g < 50 and b < 50: # Negro (Pared)
                grid[f, c] = 1
            elif r > 150 and g < 100 and b < 100: # Rojo (Inicio)
                grid[f, c] = 2
                inicio = (f, c)
            elif g > 150 and r < 100 and b < 100: # Verde (Meta)
                grid[f, c] = 3
                metas.append((f, c))
            else: # Blanco o colores claros (Camino)
                grid[f, c] = 0
                
    return grid, inicio, metas,  matriz_colores



def reconstruir_imagen(grid, tile_size=10):
    filas, columnas = grid.shape
    
    # Crear imagen vacía
    img_reconstruida = Image.new("RGB", 
                                 (columnas * tile_size, filas * tile_size),
                                 "white")
    
    for f in range(filas):
        for c in range(columnas):
            
            # Determinar color según valor del grid
            if grid[f, c] == 1:
                color = (0, 0, 0)          # Pared
            elif grid[f, c] == 2:
                color = (255, 0, 0)        # Inicio
            elif grid[f, c] == 3:
                color = (0, 255, 0)        # Meta
            elif grid[f, c] == 4:
                color = (0, 0, 255)  # Azul para el camino seguido
            else:
                color = (255, 255, 255)    # Camino
            
            # Pintar bloque
            for i in range(tile_size):
                for j in range(tile_size):
                    img_reconstruida.putpixel(
                        (c * tile_size + j, f * tile_size + i),
                        color
                    )
    return img_reconstruida

# grid, inicio, metas = procesar_laberinto("//wsl.localhost/Ubuntu/home/melmen/ia/proyecto1_IA/image.png", 10)
grid, inicio, metas, matriz_colores = procesar_laberinto("2.2.jpeg", 5) #cambiar el ancho de tiles según convenga en cada imagen

print("Grid:")
print(grid)
print("Inicio:", inicio)
print("Metas:", metas)
img = reconstruir_imagen(grid, tile_size=10)
img.show()        
img.save("debug.png")  

#Task 1.2
class Problema(ABC):

    @abstractmethod
    def inicial(self):
        pass

    @abstractmethod
    def acciones(self, estado):
        pass

    @abstractmethod
    def resultado(self, estado, accion):
        pass

    @abstractmethod
    def costo(self, estado1, accion, estado2):
        pass

    @abstractmethod
    def goalTest(self, estado):
        pass


class ProblemaLaberinto(Problema):
    def __init__(self, matriz, inicio, finales, matriz_colores):
        self.matriz = matriz
        self.inicio = inicio
        self.finales = finales
        self.filas = len(matriz)
        self.columnas = len(matriz[0])
        self.matriz_colores = matriz_colores

    def inicial(self):
        return self.inicio

    def acciones(self, estado):
        acciones = []
        fila, col = estado

        movimientos = {
            "ARRIBA": (-1, 0),
            "ABAJO": (1, 0),
            "IZQUIERDA": (0, -1),
            "DERECHA": (0, 1)
        }

        for accion, (df, dc) in movimientos.items():
            nueva_fila = fila + df
            nueva_col = col + dc

            if 0 <= nueva_fila < self.filas and 0 <= nueva_col < self.columnas:
                if self.matriz[nueva_fila][nueva_col] != 1:  # 1 = pared
                    acciones.append(accion)

        return acciones

    def resultado(self, estado, accion):
        fila, col = estado

        movimientos = {
            "ARRIBA": (-1, 0),
            "ABAJO": (1, 0),
            "IZQUIERDA": (0, -1),
            "DERECHA": (0, 1)
        }

        df, dc = movimientos[accion]
        return (fila + df, col + dc)

    def costo(self, estado1, accion, estado2):
        return 1  # costo uniforme; luego se cambia según el peso (costo) de cada decision

    def goalTest(self, estado):
        return estado in self.finales


class Nodo:
    def __init__(self, estado, padre=None, accion=None, costo=0):
        self.estado = estado
        self.padre = padre
        self.accion = accion
        self.costo = costo

def reconstruir_camino(nodo):
    camino = []
    while nodo:
        camino.append(nodo.estado)
        nodo = nodo.padre
    return list(reversed(camino))
 

def graphSearch(problema, frontera, pop_func):
    nodo_inicial = Nodo(problema.inicial())

    if problema.goalTest(nodo_inicial.estado):
        return reconstruir_camino(nodo_inicial)

    frontera.append(nodo_inicial)
    
    explorados = set()
    estados_en_frontera = {nodo_inicial.estado}   

    while frontera:
        nodo = pop_func()
        estados_en_frontera.remove(nodo.estado)  

        if problema.goalTest(nodo.estado):
            return reconstruir_camino(nodo)

        explorados.add(nodo.estado)

        for accion in problema.acciones(nodo.estado):
            estado_hijo = problema.resultado(nodo.estado, accion)

            if (estado_hijo not in explorados and 
                estado_hijo not in estados_en_frontera):

                hijo = Nodo(
                    estado_hijo,
                    padre=nodo,
                    accion=accion,
                    costo=nodo.costo + problema.costo(
                        nodo.estado, accion, estado_hijo
                    )
                )

                frontera.append(hijo)
                estados_en_frontera.add(estado_hijo) 

    return None


def BFS(problema):
    frontera = deque()
    return graphSearch(problema, frontera, frontera.popleft)

def DFS(problema):
    frontera = []
    return graphSearch(problema, frontera, frontera.pop)


def dibujar_camino_en_grid(grid, camino):
    grid_copy = grid.copy()

    for f, c in camino:
        if grid_copy[f, c] == 0:
            grid_copy[f, c] = 4  #ruta

    return grid_copy




#Para probar el trazado del laberinto ; comentar todo si no se quiere probar. 
problema = ProblemaLaberinto(grid, inicio, metas, matriz_colores)

# Ejecutar BFS (cambiar a DFS para probarlo tambien)
#camino = BFS(problema)
camino = DFS(problema)

print("Camino encontrado:", camino)

if camino:
    grid_con_camino = dibujar_camino_en_grid(grid, camino)
    img = reconstruir_imagen(grid_con_camino, tile_size=10)
else:
    print("No se encontró camino.")
    img = reconstruir_imagen(grid, tile_size=10)

img.show()
img.save("resultado.png")

#Task 1.3 #busqueda A*

#para que escoja expandir el nodo con el vallor de f mas pequeño
import heapq 

class Nodo:
    def __init__(self, estado, padre=None, accion=None, costo=0, heuristica=0):
        self.estado = estado
        self.padre = padre
        self.accion = accion
        self.costo = costo #funcon g:costo real
        self.heuristica = heuristica #funcion h: estimacion de la heuristica para llegar
        self.f = costo + heuristica #funcion f:g+h

#f mas pequeño
    def __lt__(self, otro):
        return self.f < otro.f

#Heuristica Manhatan 
def heuristica_manhattan(estado, metas):
    distacia=[abs(estado[0]-m[0])+abs(estado[1]-m[1]) for m in metas]
    return min(distacia) if distacia else 0


def A_est(problema):
    h_inicial= heuristica_manhattan(problema.inicial(), problema.finales)
    nodo_inicial=Nodo(problema.inicial(), costo=0, heuristica=h_inicial)

    frontera = []
    heapq.heappush(frontera, nodo_inicial)

    explorados=set()
    costos_g={nodo_inicial.estado:0}
    while frontera:
        nodo=heapq.heappop(frontera)

        if problema.goalTest(nodo.estado):
            return reconstruir_camino(nodo)
        
        explorados.add(nodo.estado)

        for accion in problema.acciones(nodo.estado):
            estado_hijo=problema.resultado(nodo.estado, accion)
            nuevo_costo_g=nodo.costo+problema.costo(nodo.estado,accion,estado_hijo)

            if estado_hijo not in explorados:
                if estado_hijo not in costos_g or nuevo_costo_g <costos_g[estado_hijo]:
                    costos_g[estado_hijo]=nuevo_costo_g
                    h=heuristica_manhattan(estado_hijo, problema.finales)
                    hijo=Nodo(estado_hijo, nodo, accion, nuevo_costo_g, h)
                    heapq.heappush(frontera, hijo)
    return None 

print("Ejecución A*")

camino_a_estrella= A_est(problema)
if camino_a_estrella:
    print(f"Camino A* encontrado con {len(camino_a_estrella)} pasos.")
    grid_con_a_estrella = dibujar_camino_en_grid(grid, camino_a_estrella)
    img_a_estrella = reconstruir_imagen(grid_con_a_estrella,tile_size=10)
    img_a_estrella.save("resultado_1_3.png")
    img_a_estrella.show()
else:
    print("No se ha encontrado un camino con A*.")

grid, inicio,  metas, matriz_colores=procesar_laberinto("image.png", 5)
problema=ProblemaLaberinto(grid,inicio, metas, matriz_colores)

print("Calculando ruta")
camino=A_est(problema)

if camino:
    print(f"El camino optimizado encontrado, pasos:{len(camino)}")
else:
    print("No hay camino posible")