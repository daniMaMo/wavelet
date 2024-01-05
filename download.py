import numpy as np
import csv

file_name = 'AAPL.csv'

# Read the data of CSV file.

with open(file_name, 'r') as file_csv:
    lector_csv = csv.reader(file_csv)
    # Read te data into a list of lists
    data_list = [fila for fila in lector_csv]

# Convert the list of lists to a numpy array.
data_numpy = np.array(data_list[0])

# Imprimir el arreglo NumPy
print("Datos en formato NumPy:")
print(data_numpy)

# Acceder a datos específicos del arreglo NumPy
# print("\nAcceder a datos específicos:")
# print("Nombre de la primera persona:", datos_numpy[0, 0])
# print("Edad de la segunda persona:", datos_numpy[1, 1])