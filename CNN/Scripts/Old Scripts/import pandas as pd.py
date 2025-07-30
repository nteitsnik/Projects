import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
import psycopg2 
import numpy as np
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values


username = 'postgres'
password = 'asdf1234'
host = 'localhost'        # e.g., 'localhost'
port = '5432'             # default PostgreSQL port
database = 'postgres'

engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}')


df = pd.read_csv(r"C:\Users\aiane\Downloads\PARASTATIKA-2023\PARASTATIKA-2023.csv", sep =';' ,encoding='cp1253',low_memory=False)



df['TRANDATE'] = df['TRANDATE'].str.replace(r'[\u0370-\u03FF\u1F00-\u1FFF]', '', regex=True)
df['TRANDATE'] = pd.to_datetime(df['TRANDATE'].str.split().str[0],format='%d/%m/%Y')
df['FPA'] = df['FPA'].str.replace(',', '.', regex=True)
df['FPA'] = df['FPA'].astype(float)
df['YPOKEIMENH'] = df['YPOKEIMENH'].str.replace(',', '.', regex=True)
df['YPOKEIMENH'] = df['YPOKEIMENH'].astype(float)
df['PLIROTEO'] = df['PLIROTEO'].str.replace(',', '.', regex=True)
df['PLIROTEO'] = df['PLIROTEO'].astype(float)


df.to_sql('invoices', engine, if_exists='replace', index=False,
          )


df_z = pd.read_excel(r"C:\Users\aiane\Downloads\CENTRAL-Z-REPORTS-2023\CENTRAL-Z-REPORTS-2023.xlsx")
df_z['FPA']=df_z.iloc[:, 4:8].sum(axis=1) 
df_z['PLIROTEO']=df_z.iloc[:, 9:13].sum(axis=1) 
df_z['YPOKEIMENH']=df_z['PLIROTEO']-df_z['FPA']
df_z['PARASTATIKO']='ZREP'
df_z['TRANNAME']='Z REPORT'
df_z['SEIRA']=''
df_z=df_z[['HMERA_ZHTA','KATASTHMA','PARASTATIKO','TRANNAME','SEIRA','ZNO','YPOKEIMENH','FPA','PLIROTEO']]
df_z=df_z.rename(columns={
    'HMERA_ZHTA':'TRANDATE',
    'ZNO':'ARITHMOS',
    'KATASTHMA':'STOREID'
})


df3 = pd.concat([df, df_z], ignore_index=True)

df3.to_sql('invoices', engine, if_exists='replace', index=False,
          )
