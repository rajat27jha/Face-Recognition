import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# first lets read the dataset
df = pd.read_csv('face_data.csv')
# print(df.head())
labels = df['target']
pixels = df.drop(['target'], axis=1)

# we'll split this dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(pixels, labels)

# now performing PCA
pca = PCA(n_components=135).fit(x_train)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.show()

# next step is to project our training data to PCA
x_train_pca = pca.transform(x_train)

# we initialise our svc classifier and fit our training data
clf = SVC(kernel='rbf', C=1000, gamma=0.01).fit(x_train_pca, y_train)
# rbf is a kernal for nonlinear data, c is regularization parameter
# gamma determins the smoothness of curve that our classifier fits

# at last we will test and get a classification report
x_test_pca = pca.transform(x_test)

y_pred = clf.predict(x_test_pca)
print(classification_report(y_test, y_pred))
