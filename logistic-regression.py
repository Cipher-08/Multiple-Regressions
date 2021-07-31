# train a loguistic prog for stica

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = datasets.load_iris()
x = iris.data[:,3:]
y = (iris.target==2).astype(np.int)
# print(x)
# print(y)
clf = LogisticRegression()
clf.fit(x,y)
example = clf.predict(([[1.6]]))
print(example)

# Using matplotlib for visualization 
x_new = np.linspace(0,3,1000).reshape(-1,1)
# print(x_new)
x_prob = clf.predict_proba(x_new)
print(x_prob)
plt.plot(x_new,x_prob)
plt.show()
