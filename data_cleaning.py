
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

#red the data from the excel file
data = pd.read_excel("Greece_listings.xlsx", sheet_name="Listings")
data.head(10)

#set the features and target values
x = data.iloc[:, 1:-2].values
y = data.iloc[:,-1:].values

#replace NaN valuews with the mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(x[:, 0:1])
x[:, 0:1] = imputer.transform(x[:, 0:1])

#split the dataset to training and tes datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

mu = np.mean(x_train, axis=0)
sigma = np.std(x_train, axis=0)
x_norm = (x_train - mu) / sigma
x_train = x_norm

#plot the first feature vs the target
plt.style.use("ggplot")
plt.scatter(y=y_train, x=x_train[:, 0:1], marker="o", c="b")
plt.title("Price vs Square meters")
plt.xlabel("Price")
plt.ylabel("Square meters")
plt.show()

#plot the second feature vs the target
plt.scatter(y=y_train, x=x_train[:, 1:], marker="o", c="b")
plt.title("Price vs Year")
plt.xlabel("Price")
plt.ylabel("Year")
plt.show()





