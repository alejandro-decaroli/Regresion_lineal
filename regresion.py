# Librerias
import numpy as np
import matplotlib.pyplot as plt

# Datos
x = np.random.randint(1,10,10)
y = np.random.rand(10)*2 + x

# Gráficos
plt.scatter(x,y)
plt.xlabel("Datos de x")
plt.ylabel("Datos de y")
plt.show()

# Regresión
suma = 0
for i in range(len(y)):
    suma = i[y] + suma
