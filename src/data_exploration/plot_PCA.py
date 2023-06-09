import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

X = pd.read_csv(r'./data/X_expr.csv').drop(['Unnamed: 0', 'seqLibID'], axis=1)
y = pd.read_csv(r'./data/y_cog.csv').drop(['Unnamed: 0', 'seqLibID'], axis=1)
factor = X.quantile(q=0.75, axis=1, numeric_only=True, interpolation='linear')
X = X.divide(factor, axis='rows', level=None, fill_value=None)

y_copy = y.copy()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y.values.ravel())
X[X < 0] = 0.0
X = X.loc[(X.sum(axis=1) != 0), (X.sum(axis=0) != 0)]  # select sum !=0 rows and columns


pca = PCA()
pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
pc = pipe.fit_transform(X)
plot = plt.scatter(pc[:, 0], pc[:, 1], c=y)
plt.legend(handles=plot.legend_elements()[0], labels=sorted(list(set(y_copy['DiagnosisByCognition']))))
plt.show()

# plot 3D PCA
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=y)
# plt.show()

# ################### For further inspection ###################################
remove_samples = [257, 130, 132, 5, 262, 135, 263, 265, 266, 140, 271, 16, 18, 150, 23, 29, 286, 287, 288, 36, 37,
                 166, 41, 42, 172, 191, 197, 201, 75, 203, 204, 205, 212, 214, 217, 219, 120, 93, 225, 99, 227,
                 101, 235, 241, 115, 243, 244, 245, 119, 248, 126]
outliers = [18, 132, 172]

# plot PCA without outliers
X = X.drop(outliers, axis=0)
y = y.drop(outliers, axis=0)

# plot PCA without MildCognitiveImpairment group
X = X.loc[(y['DiagnosisByCognition'] != 'MildCognitiveImpairment')]
y = y.loc[(y['DiagnosisByCognition'] != 'MildCognitiveImpairment')]

# Assigning MildCognitiveImpairment to AD group to plot PCA
y[(y['DiagnosisByCognition'] == 'MildCognitiveImpairment')] = 'AD'
