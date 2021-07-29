import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
# print(diabetes.data)
diab= diabetes.data#[:,np.newaxis,2]
# print(diab)
diab_train = diab[:-30]# from 0 to last 30
diab_test=diab[-20:]# from last 20 to last
tar = diabetes.target# taking as a feature


diab_y_train = tar[:-30]# y-axis taking with same slicing
diab_y_test = tar[-20:]# for testing

model = linear_model.LinearRegression()# making a model for plotting

model.fit(diab_train,diab_y_train)#plotting the data which we got from training data of data
diab_y_predicted = model.predict(diab_test)#this is the predicted data which we have given to it
print(mean_squared_error(diab_y_test,diab_y_predicted))

print("weights",model.coef_)
print("intercepts : ",model.intercept_)

# plt.scatter(diab_train,diab_y_train)
# plt.plot(diab_test,diab_y_predicted)
# plt.show()
