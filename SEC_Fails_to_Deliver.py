
from icecream import ic
import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor

# Load NSCC Fails-to-Deliver Data for Q1 2023
    # https://catalog.data.gov/dataset/fails-to-deliver-data
d12a = pd.read_csv(r"C:\Data\cnsfails202212a.txt", sep="|")
d12b = pd.read_csv(r"C:\Data\cnsfails202212b.txt", sep="|")
d1a = pd.read_csv(r"C:\Data\cnsfails202301a.txt", sep="|")
d1b = pd.read_csv(r"C:\Data\cnsfails202301b.txt", sep="|")
d2a = pd.read_csv(r"C:\Data\cnsfails202302a.txt", sep="|")
d2b = pd.read_csv(r"C:\Data\cnsfails202302b.txt", sep="|")
d3a = pd.read_csv(r"C:\Data\cnsfails202303a.txt", sep="|")
d3b = pd.read_csv(r"C:\Data\cnsfails202303b.txt", sep="|")

d12a = d12a.iloc[:-2]
d12b = d12b.iloc[:-2]
d1a = d1a.iloc[:-2]
d1b = d1b.iloc[:-2]
d2a = d2a.iloc[:-2]
d2b = d2b.iloc[:-2]
d3a = d3a.iloc[:-2]
d3b = d3b.iloc[:-2]

d_fails = pd.concat([d1a,d1b,d2a,d2b,d3a,d3b])
#ic(d.tail())
# ic(list(d_fails.columns))

# Determine Symbols with 31 of 62 days fails to deliver
SYMBOL_q_d = d_fails.groupby('SYMBOL')['QUANTITY (FAILS)'].agg(['sum','count'])
SYMBOL_q_d.columns = ['QUANTITY','DAYS']
SYMBOL_q_d = SYMBOL_q_d.sort_values(by=['QUANTITY'], ascending = False)
SYMBOL_q_d = SYMBOL_q_d[SYMBOL_q_d['DAYS'] == 31] # 62 trading days in Q1 2023
SYMBOL_q_d.reset_index(inplace=True)
# ic(SYMBOL_q_d)

# Append Dec 2022 data for lag terms
d_fails_ext = pd.concat([d12a,d12b,d_fails])

# Load Metrics by Individual Security Data for Q1 2023
    # https://catalog.data.gov/dataset/metrics-by-individual-security
dq123 = pd.read_csv(r"C:\Data\q1_2023_all.txt", sep="\t")
dq422 = pd.read_csv(r"C:\Data\q4_2022_all.txt", sep="\t")
d_metrics = dq123
d_metrics_ext = pd.concat([dq422,dq123])
# ic(list(d_metrics.columns))
# ic(d_metrics)

# Filter and combine datasets for selected securities
    # 25th, 50th, 75th percentile by QUANTITY
SYMBOL_q_d['Percentile Rank'] = SYMBOL_q_d.QUANTITY.rank(pct=True).round(2)
SYMBOL_25 = SYMBOL_q_d['SYMBOL'][SYMBOL_q_d['Percentile Rank'] == 0.25].iloc[0]
SYMBOL_50 = SYMBOL_q_d['SYMBOL'][SYMBOL_q_d['Percentile Rank'] == 0.5].iloc[0]
SYMBOL_75 = SYMBOL_q_d['SYMBOL'][SYMBOL_q_d['Percentile Rank'] == 0.75].iloc[0]
SYMBOL = SYMBOL_25

x = d_metrics[d_metrics['Ticker'] == SYMBOL]
y = d_fails[d_fails['SYMBOL'] == SYMBOL]
y = y.astype({'SETTLEMENT DATE':'int'})
Dc = x.merge(y,how='outer',left_on='Date',right_on='SETTLEMENT DATE')

Dc['isFail'] = np.where(Dc['QUANTITY (FAILS)'] > 0,'Fail','Good')
Dc_dates = Dc['Date']
D = Dc.drop(['Security','Ticker','SETTLEMENT DATE','CUSIP','SYMBOL','QUANTITY (FAILS)','DESCRIPTION','PRICE'], axis=1)
Dc = Dc.drop(['Date','Security','Ticker','SETTLEMENT DATE','CUSIP','SYMBOL','QUANTITY (FAILS)','DESCRIPTION','PRICE'], axis=1)
# print(Dc.to_string())
# ic(list(D.columns))

# ***************
# Base Case
# ***************

# Choose indices for train test split
np.random.seed(0)
perc = 0.85
perm = np.random.permutation(Dc.shape[0])

# # Classification using AutoGluon
# train = Dc.iloc[perm[0:round(perc*Dc.shape[0])]]
# test = Dc.iloc[perm[round(perc*Dc.shape[0]):Dc.shape[0]]]
# predictor = TabularPredictor(label='isFail').fit(train_data=train)

# # predictor = TabularPredictor.load("AutogluonModels\\ag-20230821_044833\\")
# # targets = test['isFail']
# # ic(targets)
# # predictions = predictor.predict(test)
# # ic(predictions)

# ***************
# Case 1: Velocity
# ***************

d_metrics_ext = d_metrics_ext[d_metrics_ext['Ticker'] == SYMBOL]
d_metrics_ext = d_metrics_ext[d_metrics_ext['Date'] > 20221201]
d_metrics_ext = d_metrics_ext.drop(['Security','Ticker'],axis=1)
# ic(list(d_metrics_ext.columns))
# print(d_metrics_ext.to_string())

d_metrics_1d = d_metrics_ext.iloc[: , 1:].diff(axis=0)
d_metrics_1d.insert(0,'Date',d_metrics_1d.insert)
d_metrics_1d['Date'] = d_metrics_ext.Date
d_metrics_1d = d_metrics_1d.iloc[1: , :]
d_metrics_1d = d_metrics_1d.add_suffix('_1d')
# print(d_metrics_1d.to_string())
# ic(d_metrics_1d)
# ic(list(d_metrics_1d.columns))

D_1d = d_metrics_1d.merge(D,how='inner',left_on='Date_1d',right_on='Date')
DD = D_1d
D_1d = D_1d.drop(['Date'], axis=1)

# Classification using AutoGluon
train = D_1d.iloc[perm[0:round(perc*D_1d.shape[0])]]
test = D_1d.iloc[perm[round(perc*D_1d.shape[0]):D_1d.shape[0]]]

# predictor = TabularPredictor(label='isFail').fit(train_data=train)

# predictor = TabularPredictor.load("AutogluonModels\\ag-20230828_022632\\")
# targets = test['isFail']
# ic(targets)
# predictions = predictor.predict(test)
# ic(predictions)

# ***************
# Case 2: Velocity and Acceleration
# ***************

d_metrics_2d = d_metrics_ext.iloc[: , 1:].diff(axis=0, periods=2)
d_metrics_2d.insert(0,'Date',d_metrics_2d.insert)
d_metrics_2d['Date'] = d_metrics_ext.Date
d_metrics_2d = d_metrics_2d.iloc[2: , :]
d_metrics_2d = d_metrics_2d.add_suffix('_2d')
# print(d_metrics_2d.to_string())

D_2d = d_metrics_2d.merge(DD,how='inner',left_on='Date_2d',right_on='Date')
D_2d = D_2d.drop(['Date','Date_1d'], axis=1)
# print(D_2d.to_string())

# Classification using AutoGluon
train = D_2d.iloc[perm[0:round(perc*D_2d.shape[0])]]
test = D_2d.iloc[perm[round(perc*D_2d.shape[0]):D_2d.shape[0]]]

predictor = TabularPredictor(label='isFail').fit(train_data=train)

# predictor = TabularPredictor.load("AutogluonModels\\ag-20230828_022632\\")
# targets = test['isFail']
# ic(targets)
# predictions = predictor.predict(test)
# ic(predictions)

# ***************
# Case 3: Lagged terms
# ***************




