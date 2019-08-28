#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score



#Initial dataset observations
df = pd.read_csv('Dataset/winequality-red.csv')

#Vẽ ra thể hiện số lượng sản phẩm theo từng thông số chất lượng
def plot_wine_quality_histogram(quality):
    #chọn ra những cột unique và xếp lại theo thật tự
    unique_vals = df['quality'].sort_values().unique()
    plt.xlabel("Quality")
    plt.ylabel("Count")
    plt.hist(quality.values, bins=np.append(unique_vals, 9), align='left')
    plt.show()
plot_wine_quality_histogram(df['quality'])


#--------------Data preprocessing---------------

#y: label ( cột quality )
y = df.quality
#X: dataset bỏ cột quality
X = df.drop('quality', axis=1)

#tách bộ dữ liệu ra thành training (80%) and test (20%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

#scale dữ liệu train 
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#======================Regularization============================
from sklearn.linear_model import ElasticNetCV
#n_alphas (int) số lượng số alphas trong quá trình regularization, được sử dụng cho mỗi l1_ratio
n_alphas = 300
#float giữa 0 và 1 truyền vào ElasticNet (scaling giữa L1 và L2 penalties)
l1_ratio = [.1, .3, .5, .7, .9]
#cv:số lượng k tập training được chia ra thành tập con => K-fold cross validation
Model = ElasticNetCV(n_alphas=n_alphas, l1_ratio=l1_ratio, cv=10, random_state=0)

Model.fit(X_train_scaled, y_train)

y_pred_train = Model.predict(X_train_scaled)
y_pred_test = Model.predict(X_test_scaled)
#metrics_en đánh giá giữa label có sẵn và label test đã được tách ra trước khi train
metrics = [accuracy_score(y_test, np.round(y_pred_test)),
             mean_squared_error(y_test, y_pred_test),
             r2_score(y_test, y_pred_test)]

#kết quả
result = pd.DataFrame(list(zip(metrics)), columns = ['Elastic Net Regression'],
                          index = ['acc','mse','r2'])

print(result)
