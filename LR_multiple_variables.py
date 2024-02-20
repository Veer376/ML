import pandas as p
import numpy as n

temperature_data = [20.5, 22.3, 19.8, 21.2, 20.0, -999, 23.5, 24.8, 25.5, 26.1, 20.9, 19.3, 21.8, 22.0, -999, 25.0, 27.5, 28.0, 29.2, 30.0]
sales_data = [1200, 1500, 1100, 1350, 1400, -1000, 1600, 1700, 1800, 1850, 1250, 1300, 1450, 1550, None, 1750, 1900, 2000, 2100, 2200]
steps_data = [8000, 8500, 8200, 8300, 8400, -1, 8700, 8800, 9000, 9200, 8300, 8200, 8500, None, 9100, 9300, 9400, 9500, 9600, 9700]
df = p.DataFrame({
    'Temperature': temperature_data,
    'Sales': sales_data,
    'Steps': steps_data
})

df[df<0]=n.nan
df.fillna(df.mean(), inplace=True)

from sklearn.linear_model import LinearRegression

model=LinearRegression().fit(df[['Temperature','Steps']], df['Sales']) #feature must be 2 dimentional where each column is the feature
prediction=model.predict(df[['Temperature', 'Steps']])
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


print(model.predict([[20.5, 8000]])) #this also takes up the 2d data structrue

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the data points
ax.scatter(df['Temperature'], df['Steps'], df['Sales'], c='r', marker='o', label='Data Points')

# Linear regression line
min_temp = df['Temperature'].min()
max_temp = df['Temperature'].max()
min_steps = df['Steps'].min()
max_steps = df['Steps'].max()
min_sales = model.predict([[min_temp, min_steps]])[0]  # Extract scalar value
max_sales = model.predict([[max_temp, max_steps]])[0]  # Extract scalar value
ax.plot([min_temp, max_temp], [min_steps, max_steps], [min_sales, max_sales], color='b', label='Linear Regression Line')

plt.show()
