
#Importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'Datasets')
print(os.listdir(CONFIG_PATH))



data = pd.read_csv(CONFIG_PATH + '/Social_Network_Ads.csv')
#print(data)
data.drop(columns=['User ID','Gender',],axis=1,inplace=True)
data.head()

#Khai báo label là cột cuối cùng trong tệp nguồn
y = data.iloc[:,-1].values

# Khai báo X (features) là tất cả các cột không bao gồm cuối cùng
X = data.iloc[:,:-1].values

# Phân tách dữ liệu
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#print(X_train)

# Phân tách dữ liệu
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import the model
from sklearn.linear_model import LogisticRegression
# Make an instance of the Model
classifierLR = LogisticRegression()
# Training the model on the data
# Model is learning the relationship between digits (x_train) and labels (y_train)
classifierLR.fit(X_train, y_train)

# Find Accuracy of training data
from sklearn.metrics import accuracy_score
    # predict training data
#y_pred_train=classifierLR.predict(X_train)
#print(accuracy_score(y_train, y_pred_train))
    #predicting the test label with LR. Predict always takes X as input
    # Predict labels for new data (new images)
y_predLR=classifierLR.predict(X_test)
print(accuracy_score(y_test, y_predLR))



# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifierLR.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

## Visualising the Test set results
#from matplotlib.colors import ListedColormap
#X_set, y_set = X_test, y_test
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifierLR.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Logistic Regression (Test set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()