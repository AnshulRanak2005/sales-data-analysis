import pandas as pd

# Load dataset
data = pd.read_csv("Superstore.csv")

# Show data
print(data.head())

# Basic info
print(data.info())

# Summary
print(data.describe())
import matplotlib.pyplot as plt

# Sales by Category
data.groupby('Category')['Sales'].sum().plot(kind='bar')

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Select features
x = data[['Sales', 'Quantity', 'Discount']]
y = data['Profit']

# Split data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("Predictions:", predictions[:5])