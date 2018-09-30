
import matplotlib.pyplot as plt
import pandas as pd

datos = pd.read_csv("Position_Salaries.csv")
X = datos.iloc[:, 1:2].values     
y = datos.iloc[:, 2].values 
#ajuste
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
Reg_poli= PolynomialFeatures(degree = 4)
X_poly = Reg_poli.fit_transform(X)
Reg_line = LinearRegression()
Reg_line.fit(X_poly, y)

#plotear
plt.scatter(X, y, color = 'red')
plt.plot(X, Reg_line.predict(X_poly), color = 'blue')
plt.title('Regresion Polinomial')
plt.xlabel('Nivel de Posicion')
plt.ylabel('Salario')
plt.show()

# Prediccion
salario = Reg_line.predict(Reg_poli.fit_transform(8.5))
print("PODEMOS OFRECERLE ", salario,"DE SALARIO")
