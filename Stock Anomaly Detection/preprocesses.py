import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("archive/indexData.csv")
df.drop(columns=["Index"], inplace=True)

date = pd.to_datetime(df["Date"])

df["Date"] = date

open = df["Open"]
mean = open.mean()
mean = round(mean, 2)
df["Open"].fillna(mean, inplace=True)
for i in range(len(open)):
    if type(open[i]) == str:
        open[i] = mean
df["Open"] = open

scaler = StandardScaler()
open_scaler = scaler.fit_transform(pd.DataFrame(df["Open"]))
df["Open"] = open_scaler

high = df["High"]
mean = high.mean()
mean = round(mean, 2)
df["High"].fillna(mean, inplace=True)
for i in range(len(high)):
    if type(high[i]) == str:
        high[i] = mean

df["High"] = high

scaler = StandardScaler()
high_scaler = scaler.fit_transform(pd.DataFrame(df["High"]))
df["High"] = high_scaler

low = df["Low"]
mean = low.mean()
mean = round(mean, 2)
df["Low"].fillna(mean, inplace=True)
for i in range(len(low)):
    if type(low[i]) == str:
        low[i] = mean
df["Low"] = low

scaler = StandardScaler()
low_scaler = scaler.fit_transform(pd.DataFrame(df["Low"]))
df["Low"] = low_scaler

close = df["Close"]
mean = close.mean()
mean = round(mean, 2)
df["Close"].fillna(mean, inplace=True)
for i in range(len(close)):
    if type(close[i]) == str:
        close[i] = mean

scaler = StandardScaler()
close_scaler = scaler.fit_transform(pd.DataFrame(df["Close"]))
df["Close"] = close_scaler

adj_close = df["Adj Close"]
mean = adj_close.mean()
mean = round(mean, 2)
df["Adj Close"].fillna(mean, inplace=True)
for i in range(len(adj_close)):
    if type(adj_close[i]) == str:
        adj_close[i] = mean

adj_close_scaler = scaler.fit_transform(pd.DataFrame(df["Adj Close"]))
df["Adj Close"] = adj_close_scaler
# standard_cols = ["Open", "High", "Low", "Close", "Adj Close"]

# for col in standard_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")
#     df[col].fillna(df[col].mean(), inplace=True)
# scaler = StandardScaler()
# adj_close_scaler = scaler.fit_transform(pd.DataFrame(df["Adj Close"]))
# df["Adj Close"] = adj_close_scaler

volume = df["Volume"]
mean = volume.mean()
mean = round(mean, 2)
for i in range(len(volume)):
    if type(volume[i]) == str:
        volume[i] = mean
scaler = MinMaxScaler(feature_range=(0, 10000))
volume_scaler = scaler.fit_transform(pd.DataFrame(df["Volume"]))
df["Volume"] = volume_scaler


df.to_csv("preprocessed_data.csv", index=True)