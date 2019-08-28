
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.externals import joblib
from sklearn import preprocessing
from keras.utils.vis_utils import plot_model
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)

dataset = pd.read_csv('Datasets\heart.csv')
# split into input (X) and output (Y) variables
X = dataset.iloc[:,0:13]
Y = dataset.iloc[:,13]

#scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X)
#print(X_train)

# create model
model = Sequential()
model.add(Dense(16, input_dim=13, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=250, batch_size=20)

# evaluate the model
scores = model.evaluate(X, Y)
print("scores: \n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

##Saving model 
#filename = 'NeuralNetwork_CancerModel.pkl'
#joblib.dump(model,filename)
#Load model
#NeuralNetwork_CancerModel = joblib.load("NeuralNetwork_CancerModel.pkl")

#DataInput=pd.DataFrame([[58],[1],[0],[150],[270],[0],[0],[111],[1],[0.6],[2],[0],[2]]).T
#DataInput.columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
#print(DataInput)
# calculate predictions

# round predictions
predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]
#the result of prediction 
print(rounded)


