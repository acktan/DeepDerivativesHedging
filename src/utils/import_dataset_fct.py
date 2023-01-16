import pandas as pd
import os 

def get_file(filename):
    df =  pd.read_csv("/home/jovyan/hfactory_magic_folders/natixis_data_challenge_22_23/erm/" + filename,                     header=None)
    return df

def return_filenames():
    return os.listdir("/home/jovyan/hfactory_magic_folders/natixis_data_challenge_22_23/erm")