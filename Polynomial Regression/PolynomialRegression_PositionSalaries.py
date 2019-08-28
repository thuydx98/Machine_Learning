import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'Datasets')
data = pd.read_csv(CONFIG_PATH + '/Position_Salaries.csv')

X = data.iloc[:,1:2].values
y = data.iloc[:,-1].values

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
print(X_poly)

# linear regression model
from sklearn.linear_model import LinearRegression
linear_reg_model = LinearRegression()
linear_reg_model.fit(X, y)

# polynomial regression model
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_poly, y)

#from sklearn.metrics import accuracy_score
#y_predLR = poly_reg_model.predict(poly_reg.fit_transform(X));
#print(accuracy_score(y, y_predLR))

import matplotlib.pyplot as plt
plt.scatter(X, y, color='red', label='Actual observation points')
plt.plot(X, linear_reg_model.predict(X), label='Linear regressor fit curve')
plt.plot(X, poly_reg_model.predict(poly_reg.fit_transform(X)), label='Polynmial regressor fit line')
plt.title('Truth or bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')

plt.legend()
plt.show()
