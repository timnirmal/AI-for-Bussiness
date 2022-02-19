# Load CSV
import pandas as pd

results = pd.read_csv('segmentation_results.csv')

print(results.head())

# Plot the results
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the results
plt.figure(figsize=(10, 6))
sns.barplot(x='RFM_Score', y='CustomerID', data=results)
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RFM_Score', y='CustomerID', data=results)
plt.show()

# Another plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='RFM_Score', y='CustomerID', hue='RFM_Score', data=results)  # , palette='coolwarm')
# https://seaborn.pydata.org/generated/seaborn.scatterplot.html
plt.show()
