# IMPORT LIBRARIES

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# READING DATA FILES
df = pd.read_csv('Combine.csv')

# PRINTING HEAD
print(df.head())

# CHECK FOR NULL VALUES
print(sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis'))

# DROPPING NA VALUES
df = df.dropna()

# INDEPENDENT FEATURES
X = df.iloc[:, :-1]

# DEPENDENT FEATURES
y = df.iloc[:, -1]

# CHECK NULL VALUES
print(X.isnull())
print(y.isnull())

# PLOTTING GRAPH WITH EVERY POSSIBLE PAIRS
print(sns.pairplot(df))

# CORRELATION BETWEEN THE PAIRS
'''
Correlation states how the features are related to each other or the target variable.
Correlation can be positive (increase in one value of feature increases the value of the target variable)
or negative (increase in one value of feature decreases the value of the target variable)Heatmap makes it
easy to identify which features are most related to the target variable, we will plot heatmap of correlated 
features using the seaborn library
'''
print(df.corr())

# GET CORRELATION OF EACH FEATURES IN DATASET
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))

# PLOT HEAT MAP
g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
print(corrmat.index)

'''
FEATURE IMPORTANCE GIVES A SCORE FOR EACH FEATURE OF DATA HIGHER THE SCORE MORE THE IMPORTANT TOWARDS DEPENDENT VARIABLE
FEATURE IMPORTANCE IS INBUILT CLASS THAT COMES WITH TREE BASE BASED REGERESSOR , WILL BE EXTRA TREE REGRESSOR FOR EXTRACTING 
THE TOP 10 FEATURES OF DATASET
'''
from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor()
model.fit(X, y)
print(X.head())
print(model.feature_importances_)

# FEATURE IMPORTANT FOR VISUALIZATION
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()

# DISPLOT REPRESENSTS THE UNIVARIATE DISTRIBUTION OF DATA ,DATA DISTRIBUTION OF A VARIABLE AGAINST THE DENSITY
# DISTRIBUTION
sns.distplot(y)

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# TRAINING OF MODEL
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# COEFFICIENT'S VALUES
print(regressor.coef_)
print(regressor.intercept_)
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_train, y_train)))
print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))
from sklearn.model_selection import cross_val_score

score = cross_val_score(regressor, X, y, cv=5)
print(score.mean())

# EVALUATION
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
prediction = regressor.predict(X_test)

# PLOTING Y-TEST-PREDICTION
sns.distplot(y_test - prediction)
plt.scatter(y_test, prediction)

# MEAN SQUARED ERROR , MEAN ABSOLUTE ERROR, ROOT MEAN SQUARE ERROR
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

# BUILDING MODEL
import pickle

# OPENING FILE WHERE MODEL WILL BE STORED IN WRITE BINARY MODE
file = open('linear_model.pkl', 'wb')
# DUMPING MODEL TO THE FILE
pickle.dump(regressor, file)
