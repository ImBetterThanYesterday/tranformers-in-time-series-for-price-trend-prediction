import numpy as np
import pandas as pd

# Ruta al archivo .npy (ajústala a tu caso)
# path_npy = "data/BTC/individual_examples/example_1.npy"

path_npy = "data/BTC/test.npy"

# 1. Cargar el array
arr = np.load(path_npy)   # si te da error, prueba: np.load(path_npy, allow_pickle=True)

print(arr.shape)  # para ver la dimensión y saber cómo tratarlo

# 2. Convertir a DataFrame según la forma

if arr.ndim == 1:
    # Vector
    df = pd.DataFrame(arr, columns=["value"])

elif arr.ndim == 2:
    # Matriz (filas x columnas)
    df = pd.DataFrame(arr)

elif arr.ndim == 3:
    # Por ejemplo (N, 50, 40) -> aplastamos las últimas dimensiones
    n_samples = arr.shape[0]
    df = pd.DataFrame(arr.reshape(n_samples, -1))
else:
    raise ValueError(f"No sé bien cómo tabular un array con {arr.ndim} dimensiones")

# 3. Guardar en Excel
output_path = "example_12u.xlsx"
df.to_excel(output_path, index=False)

print(f"Guardado en: {output_path}")
