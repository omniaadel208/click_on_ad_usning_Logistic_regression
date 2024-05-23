# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# %%
advertising = pd.read_csv('advertising.csv')

# %%
advertising.head()

# %%
advertising.isnull().sum()

# %%
advertising['Ad Topic Line'].unique()

# %%
advertising['Country'].unique()

# %%
advertising['City'].unique()

# %%
advertising.info()

# %%
advertising.describe()

# %%
sns.histplot(data=advertising, x='Age')

# %%
sns.jointplot(data=advertising, x='Age', y='Area Income')

# %%
sns.jointplot(data=advertising, x='Age', y='Daily Time Spent on Site', kind='kde')
plt.show()

# %%
sns.jointplot(data=advertising, x='Daily Time Spent on Site', y='Daily Internet Usage', color='green')

# %%
sns.pairplot(advertising, hue='Clicked on Ad')

# %%
y = advertising['Clicked on Ad']

# %%
y.head()

# %%
advertising.drop(columns='Clicked on Ad', inplace=True)

# %%
advertising.drop(columns=['Ad Topic Line', 'City', 'Country', 'Timestamp'], inplace=True)

# %%
x = advertising.copy()

# %%
x.head()

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %%
lr_model = LogisticRegression()

# %%
lr_model.fit(x_train, y_train)

# %%
y_pred = lr_model.predict(x_test)

# %%
print(classification_report(y_test, y_pred))



# %%
