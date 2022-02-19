# https://towardsdatascience.com/product-segmentation-for-retail-with-python-c85cc0930f9a
# https://github.com/samirsaci/product-segmentation/blob/main/Product%20Segmentation%20with%20Statistics.ipynb


# Product segmentation refers to the activity of grouping products that have similar characteristics and serve a
# similar market.

# You want to put efforts into managing products that have:
# The highest contribution to your total turnover: ABC Analysis
# The most unstable demand: Demand Variability

# In this article, we will introduce simple statistical tools to combine ABC Analysis and Demand Variability to
# perform products segmentation.


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

PATH = "m5-forecasting-accuracy/"

##################################### 1. Data Preparation

# -- Parameters
# Columns
COLS_ITM = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
# Take the first year data
COLS_DATE = ['d_' + str(i) for i in range(1, 366)]

# Import Sales Data
df = pd.read_csv(PATH + 'sales_train_evaluation.csv')
# Change id to match with price dataframe
df['id'] = df['store_id'] + '-' + df['item_id']
# Set index
df.set_index(COLS_ITM, inplace=True)
# Scope = Year 1
df = df[COLS_DATE]

# Import Calendar
df_calendar = pd.read_csv(PATH + 'calendar.csv')
# Date to Week
dict_week = dict(zip(df_calendar.d.values, df_calendar.wm_yr_wk.values))

# Import Pricing
df_price = pd.read_csv(PATH + 'sell_prices.csv')
# SKU Index
df_price['item_store_id'] = df_price['store_id'] + '-' + df_price['item_id']
# Pricing
df_price = df_price.pivot(index='item_store_id', columns='wm_yr_wk', values='sell_price').fillna(0)
# Matrix from pivot
matrix_price = df_price.to_numpy()
# Dict Matrix Index
pr_n = dict(zip(df_price.index, range(len(df_price.index))))
pr_p = dict(zip(df_price.columns, range(len(df_price.columns))))

print(df_price.head())
print(df_price.shape)
print(df_price.columns)
print(df_price.index)

# Filter on the first year of sales for HOBBIES Skus
# Calculate Mean, Standard deviation and CV of sales
# Sorting (Descending) and Cumulative sales calculation for ABC analysis

"""
Products Rotation
What are the references that are driving most of your sales?
Very Fast Movers: top 5% (Class A)
The following 15% of fast movers (Class B)
The remaining 80% of very slow movers (Class C)
This classification will impact,
Warehouse Layout: Reduce Warehouse Space with the Pareto Principle using Python, Samir Saci, Link
Picking Process: Improve Warehouse Productivity using Order Batching with Python, Samir Saci, Link
Demand Variability
How stable is your customers’ demand?
Average Sales: µ
Standard Deviation:
Coefficient of Variation: CV = σ/µ
"""

#####################################

# ABC SKU-LEVEL
df_abc = df[df['cat_id'] == 'HOBBIES'].copy()
df_abc = df_abc.groupby(['item_id', 'dept_id', 'cat_id']).sum()

# Calculate mean and standard deviation
# Mean
df_abc['mean'] = df_abc[COLS_DATE].mean(axis=1)
# Filter out the reference without sales
df_abc = df_abc[df_abc['mean'] > 0]
# Standard Deviation
df_abc['std'] = df_abc[COLS_DATE].std(axis=1)
# Coefficient of Variation
df_abc['CV'] = df_abc['std'] / df_abc['mean']
df_abc.reset_index(inplace=True)
# # ABC SKU-LEVEL
df_abc = df_abc.drop(COLS_DATE, axis=1).copy()
df_abc['TO%'] = (100 * df_abc['TO'] / df_abc['TO'].sum())
# Sort
df_abc.sort_values(['TO%'], ascending=False, inplace=True, ignore_index=True)
df_abc['TO%_CS'] = df_abc['TO%'].cumsum()

# A, B, C on SKU Number
n_sku = len(df_abc)
n_a, n_b = int(0.05 * n_sku), int(0.5 * n_sku)
df_abc['SKU_ID'] = pd.Series(range(1, len(df_abc))).astype(int)
df_abc['SKU_%'] = (100 * pd.Series(range(1, len(df_abc))) / len(df_abc))
df_abc['ABC'] = pd.Series(range(len(df_abc))).apply(lambda t: 'A' if t <= n_a - 1 else 'B' if t <= n_b - 1 else 'C')
# A, B, C on turnover
to_a, to_b = df_abc[df_abc['SKU_ID'] == n_a]['TO%'].max(), df_abc[df_abc['SKU_ID'] == n_b]['TO%'].max()

##################################### 2. ABC Analysis

# Distribution by Value
ax = plt.gca()
df_abc.plot(figsize=(12, 8), x='SKU_%', y='TO%_CS', ax=ax, grid=True)
# ABC
# 20%, 50% of SKU Number
ax.axvline(5, color="red", linestyle="-", linewidth=1.0)
ax.axvline(20, color="red", linestyle="-", linewidth=1.0)
# 20%, 50% of SKU Number
ax.axhline(80, color="blue", linestyle="--", linewidth=1.0)
ax.axhline(95, color="blue", linestyle="--", linewidth=1.0)
plt.xlabel('Percentage of SKU (%)')
plt.xticks(rotation=90)
plt.ylabel('Percentage of the Annual Turnover (%)')
plt.title('ABC Analysis: Distribution by Turnover (Sales Value in $)')
plt.show()

##################################### 2. Demand Variability

# Distribution by Value
ax = plt.gca()
df_abc.plot(figsize=(12, 8), x='SKU_%', y='TO%_CS', ax=ax, grid=True)
# ABC
# 20%, 50% of SKU Number
ax.axvline(5, color="red", linestyle="-", linewidth=1.0)
ax.axvline(20, color="red", linestyle="-", linewidth=1.0)
# 20%, 50% of SKU Number
ax.axhline(80, color="blue", linestyle="--", linewidth=1.0)
ax.axhline(95, color="blue", linestyle="--", linewidth=1.0)
plt.xlabel('Percentage of SKU (%)')
plt.xticks(rotation=90)
plt.ylabel('Percentage of the Annual Turnover (%)')
plt.title('ABC Analysis: Distribution by Turnover (Sales Value in $)')
plt.show()

##################################### 3. Normality Test

# Bar Chart
ax = plt.gca()
colors = {'A': 'red', 'B': 'green', 'C': 'blue'}
# Remove Outliers
df_plot = df_abc[df_abc['CV'] < 4].copy()
df_plot.plot.scatter(figsize=(12, 8), x='TO%', y='CV', color=df_plot['ABC'].map(colors), ax=ax, grid=True)
# ABC
# A, B and C
ax.axvline(to_a, color="red", linestyle="-", linewidth=1.0)
ax.axvline(to_b, color="red", linestyle="-", linewidth=1.0)
# 20%, 50% of SKU Number
ax.axhline(1, color="blue", linestyle="--", linewidth=1.0)
plt.xlabel('Percentage of Turnover (%)')
plt.xticks(rotation=90)
plt.ylabel('Coefficient of Variation')
plt.title('Distribution by Demand Variability')
plt.show()
