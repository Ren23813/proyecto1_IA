#task 1.1 Discretización del mundo

import numpy as np
from PIL import Image

def procesar_laberinto(ruta_imagen, tile_size=10):
    # 1. Cargar imagen y convertir a RGB
    img = Image.open(ruta_imagen).convert('RGB')
    ancho, alto = img.size
    
    # Calcular dimensiones de la matriz 
    columnas = ancho // tile_size
    filas = alto // tile_size
    
    grid = np.zeros((filas, columnas), dtype=int)
    inicio = None
    metas = []

    for f in range(filas):
        for c in range(columnas):
            # Definir la caja del tile (left, top, right, bottom)
            caja = (c * tile_size, f * tile_size, (c + 1) * tile_size, (f + 1) * tile_size)
            tile = img.crop(caja)
            
            color_medio = np.array(tile).mean(axis=(0, 1))
            r, g, b = color_medio
            
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
                
    return grid, inicio, metas

