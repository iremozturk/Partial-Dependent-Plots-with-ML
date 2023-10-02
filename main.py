import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from DecisionTree.DecisionTree import plot_partial_dependence, train_decision_tree

#from sklearn.inspection import plot_partial_dependence
# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv(r'C:\Users\iremo\Downloads\outputsu.csv')
# Select numerical columns and convert to float

numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].astype(float)

# Calculate mode values for each numerical column
mode_values = df[numerical_cols].mode()

# Iterate through numerical columns and replace non-numeric values with the mode value
for col in numerical_cols:
    mode = mode_values[col].values[0]  # Get the mode value for the column
    df[col] = df[col].apply(lambda x: mode if x == '?' else x)


# Handle other missing values (e.g., replace NaNs with median)
df.fillna(df.mode(), inplace=True)

# You may need to handle missing values for other categorical columns as well
# ...

# Now, your DataFrame should only contain numeric values or appropriate replacements

# Define your target variable (binaryClass in your case)
X = df.drop(columns=['binaryClass'])
y = df['binaryClass']
decision_tree = train_decision_tree(X_train, y_train)

# Generate and plot Partial Dependence Plots (PDPs)
features_to_plot = ['age', 'TSH measured']
plot_partial_dependence(decision_tree, X_train, features=features_to_plot)