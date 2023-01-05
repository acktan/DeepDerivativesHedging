import import_dataset_fct
import pandas as pd
import numpy as np

def col_numeric_names(df):
    df = df.rename(columns={x:y for x,y in zip(df.columns,range(0,len(df.columns)))})
    return df

def absolute_growth(df):
    len_df = df.shape[1]
    temp = df.iloc[:,1:len_df]
    temp = col_numeric_names(temp)
    temp2 = df.iloc[:,:len_df-1]
    df2 = temp-temp2
    df2.insert(0, "0", 0)
    df2 = col_numeric_names(df2)
    return df2

def percentage_growth(df):
    df2 = absolute_growth(df)
    df2 = df2.drop(columns=[0])
    df2 = col_numeric_names(df2)
    df3 = df2.div(df.iloc[:,:df.shape[1]-1])
    df3.insert(0, "0", 0)
    df3 = col_numeric_names(df3)
    return df3