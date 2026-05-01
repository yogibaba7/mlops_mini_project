import pandas as pd 

df = pd.read_csv("data/interim/test_preprocessed.csv")

print(df.isnull().sum().sum())