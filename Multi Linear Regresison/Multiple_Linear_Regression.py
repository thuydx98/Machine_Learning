
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('Datasets\Startups.csv')
#bỏ cột cuối Profit
X = dataset.iloc[:, :-1]
#lấy cột 4: profit
y = dataset.iloc[:, 4]
#Convert the column into categorical columns
#chuyển trang binary vector theo index, bỏ cột  California
states=pd.get_dummies(X['State'],drop_first=True)

# Drop the 'state' coulmn
X=X.drop('State',axis=1)

# concat the dummy variables
# thêm lại cột 'states' vào X khi mà 'states' đã được xử lí sang dạng số
#axis=1 tức là chèn cột, axis=0 là chèn hàng(record ), pd.concat([gốc,dữ liệu thêm],vị trí cột/hàng)
#column name:  Index | R&D Spend | Admin | Marketing Spend | Florida | New York
X=pd.concat([X,states],axis=1)
#print(X)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#---------------After train 1: chạy dự đoán luôn kết quả -------------------------
# Predicting the Test set results
# y dự đoán ( do model dự đoán )
y_pred = regressor.predict(X_test)
# y kết quả thực tách từ dataset ra
score=r2_score(y_test,y_pred)
print('score: '+str(score))

#---------------After train 2: lưu lại model để dự đoán kết quả -------------------------
#-------------Save section-----------------------

#filename = 'LinearRegression_model.pkl'
##lưu lại: joblib.dump(model, filename)
#joblib.dump(regressor,filename)

#------------Use section--------------------
## some time later

#load the model from disk
#from sklearn.metrics import r2_score
#loaded_model = joblib.load("LinearRegression_model.pkl")
## Predicting the Test set results
## y dự đoán ( do model dự đoán )
#y_pred = loaded_model.predict(X_test)
#score=r2_score(y_test,y_pred)
#print('score: '+str(score))
