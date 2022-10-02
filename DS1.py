##pip install -r requirements.txt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
df=pd.read_csv('diamonds.csv')

print("-------------------- \n","HeAd: \n", df.head())
print("-------------------- \n","InFo: \n")
df.info()
print("-------------------- \n","DescriptiVe: \n", df.describe())
print("-------------------- \n","ShaPe: \n", df.shape)
print("-------------------- \n","Are there nulLs? \n", df.isnull().sum())

def unique():
    for column in df.columns:
        if df.dtypes[column] == 'object':
            print({column: len(df[column].unique())})
print("-------------------- \n","UniQue: \n")
unique()

fig1 = px.scatter(data_frame = df, x="carat",
                    y="price", size="depth",
                    color= "cut", trendline="ols")
fig1.show()

fig2 = px.box(df, x="cut",
             y="price",
             color="color")
fig2.show()

fig3 = px.box(df,
             x="cut",
             y="price",
             color="clarity")
fig3.show()

df['cut'].replace(['Ideal','Premium','Good','Very Good','Fair'],[0,4,2,3,1], inplace=True)
df['color'].replace(['D','E','F','G','H','I','J'],[0,1,2,3,4,5,6], inplace=True)
df['clarity'].replace(['SI1','SI2','I1','IF','VS1','VS2','VVS1','VVS2'],[0,1,2,3,4,5,6,7], inplace=True)

print("-------------------- \n","HeAd: \n", df.head())

plt.figure(figsize = (16, 10))
sns.heatmap(df.corr(), annot = True, cmap="YlGnBu")
plt.show()

df1 = np.random.rand(len(df)) < 0.8
train = df[df1]
test = df[~df1]

regr = linear_model.LinearRegression()
x = np.asanyarray(train[['carat','x','y','z']])
y = np.asanyarray(train[['price']])
regr.fit (x, y)
print ('Coefficients: ', regr.coef_)

from sklearn.metrics import r2_score

y_hat= regr.predict(test[['carat','x','y','z']])
x = np.asanyarray(test[['carat','x','y','z']])
y = np.asanyarray(test[['price']])
print("Residual sum of squares (MSE): %.2f"
      % np.mean((y-y_hat) ** 2))
print("R2-score: %.2f" % r2_score(y_hat , y) )

print('Explained Variance score: %.2f' % regr.score(x, y))