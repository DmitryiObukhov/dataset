import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

tips = sns.load_dataset('tips')

tips.head()
tips.describe()
tips.isnull().sum()
tips.hist(figsize=(10, 10))
plt.show()

sns.pairplot(tips)
plt.show()

sns.boxplot(x='total_bill', data=tips)
plt.show()

tips_cleaned = tips.dropna()
numeric_columns = tips.select_dtypes(include=[np.number]).columns
tips_filled = tips.fillna(tips[numeric_columns].mean())

tips['tip_percentage'] = (tips['tip'] / tips['total_bill']) * 100
tips.describe()

sns.histplot(tips['tip_percentage'])
plt.show()

sns.scatterplot(x='total_bill', y='tip', data=tips)

numeric_columns = tips.select_dtypes(include=[np.number]).columns
correlation_matrix = tips[numeric_columns].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

fig = px.scatter_matrix(tips, dimensions=tips.columns, color='tip')
fig.show()
