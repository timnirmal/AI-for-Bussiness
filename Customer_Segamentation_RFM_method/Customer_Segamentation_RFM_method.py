import pandas as pd

data_df = pd.read_excel("Online Retail.xlsx")

# data_df = pd.read_csv('Online_Retail.csv', encoding='unicode_escape')

print(data_df.head())

################################### 1. Preparing Data

data_df['TotalPrice'] = data_df['Quantity'].astype(int) * data_df['UnitPrice'].astype(float)
data_df['InvoiceDate'] = pd.to_datetime(data_df['InvoiceDate'])

# Now that we have our necessary columns ready, we can perform the following calculations for our recency,
# frequency and monetary values:

# Recency: Calculate the number of days between the most recent date and the last date of purchase for each customer.
# Frequency: Calculate the number of orders for each customer.
# Monetary: Calculate the sum of all the money spent for each customer.

# We can perform this calculations quickly with a lambda function:

rfm = data_df.groupby('CustomerID').agg({'InvoiceDate': lambda date: (date.max() - date.min()).days,
                                         'InvoiceNo': lambda num: len(num),
                                         'TotalPrice': lambda price: price.sum()})

# Change the name of columns
rfm.columns = ['recency', 'frequency', 'monetary']

# Computing Quantile of RFM values
rfm['recency'] = rfm['recency'].astype(int)
rfm['r_quartile'] = pd.qcut(rfm['recency'].rank(method='first'), 4, ['1', '2', '3', '4']).astype(int)
rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4', '3', '2', '1']).astype(int)
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4', '3', '2', '1']).astype(int)

# Concatenating the quantile numbers to get the RFM score
rfm['RFM_Score'] = rfm.r_quartile.astype(str) + rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)

# Sort the outcomes
print('Sorting customers')
sorted_customers = rfm.sort_values('RFM_Score', ascending=True)
sorted_customers.to_csv('segmentation_results.csv', index=True)

# https://towardsdatascience.com/automated-customer-segmentation-2f9cec9df4df
# Google Sheet - https://medium.com/game-of-data/play-with-google-spreadsheets-with-python-301dd4ee36eb
# RFM - https://www.kaggle.com/dansbecker/rfm-customer-segmentation
# https://towardsdatascience.com/know-your-customers-with-rfm-9f88f09433bc
