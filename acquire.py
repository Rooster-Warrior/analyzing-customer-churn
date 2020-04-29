from env import password, user, host
import pandas as pd
import numpy as np
import os

query = '''

SELECT * 
FROM customers
JOIN contract_types USING (contract_type_id)
JOIN payment_types USING (payment_type_id)
JOIN internet_service_types USING (internet_service_type_id);

'''

data_base_name = "telco_churn"

def sql_database(host=host, user=user, password=password):
    '''
    Function used to read query from SQL database
    ''' 
    global query
    global data_base_name

    url = f'mysql+pymysql://{user}:{password}@{host}/{data_base_name}'
    df = pd.read_sql(query, url)
    return df

def pull_csv_file():
    '''
    Function used to create a csv file from the SQL datatabase
    '''
    global data_base_name
    global query
    url = f'mysql+pymysql://{user}:{password}@{host}/{data_base_name}'
    plain_data = pd.read_sql(query, url)
    plain_data.to_csv("telco_churn_data.csv")
    

def check_for_csv_file(file_name):
    '''
    Checks if there is a csv file with the matching name in the directory. If there isn't 
    it will create a new csv using the env file in the directory. 
    '''
    if os.path.exists(file_name) == False:
        pull_csv_file()

def read_telco_data():
    ''' 
    Used to read the telco csv file. It also drops the unnamed column
    '''
    df = pd.read_csv("telco_churn_data.csv")
    df = df.drop(columns="Unnamed: 0")
    return df



