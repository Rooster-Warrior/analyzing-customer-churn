from env import password, user, host, url
import pandas as pd




def sql_database(query, data_base_name, host=host, user=user, password=password):
    query = '''

    SELECT * 
    FROM customers
    JOIN contract_types USING (contract_type_id)
    JOIN payment_types USING (payment_type_id)
    JOIN internet_service_types USING (internet_service_type_id);

    '''

    data_base_name = "telco_churn"

    url = f'mysql+pymysql://{user}:{password}@{host}/{data_base_name}'
    df = pd.read_sql(query, url)
    return df

def prep_data(df):
    df.total_charges = df.total_charges.replace(" ", df.tenure*df.monthly_charges)
    df.total_charges = df.total_charges.astype(float)
    return df