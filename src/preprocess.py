import pandas as pd
import math


data=pd.read_csv(r'../data/bengaluru_house_prices.csv')


data = data.apply(lambda x: x.str.strip().str.lower().str.replace(r'\s+', ' ', regex=True) if x.dtype == 'object' else x)
data.describe()

data[['location', 'size', 'society']] = data[['location', 'size', 'society']].fillna('unknown')

bath_average=int(data['bath'].mean())
data['bath_cleaned']=data['bath'].fillna(bath_average)
data=data.drop(columns='bath')
data=data.rename(columns={"bath_cleaned":"bath"})

balcony_average=int(data['balcony'].mean())
data['balcony_cleaned']=data['balcony'].fillna(bath_average)
data=data.drop(columns='balcony')
data=data.rename(columns={"balcony_cleaned":"balcony"})

duplicates = data[data.duplicated(keep=False)]
data.duplicated().sum()
data= data.drop_duplicates(keep='first') 
data['bath']=data['bath'].astype(int)
data['balcony'] = data['balcony'].astype(int)

