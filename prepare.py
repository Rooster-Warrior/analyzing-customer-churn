
from sklearn.model_selection import train_test_split

# ----------------------- #
#       Data Prep         #
# ----------------------- #

def replace_missing_values(df):
    df.total_charges = df.total_charges.replace(" ", df.tenure*df.monthly_charges)
    df.total_charges = df.total_charges.astype(float)
    return df

def create_tenure_year(df):
    df["tenure_years"] = df.tenure/12
    df.tenure_years = df.tenure_years.astype(int)
    return df

# We had two options to deal with categorical data that had a n/a result (no phone service or no internet service), we chose to encode this as a seperate option (value=2) rather than lump in with other negative i.e. "No" values.
#  This could have implications when we are modeling because there is no value relationship between the integers used (e.g. 2 is not more than 1)
def encode_all(df):
    """encodes all Yes values to 1, No values to 0, and 2 for n/a results of no internet service
    or no phone service,  Female to 1 and Male to 0 then turns encode columns into integers"""
    df = df.replace({"Yes": 1,
                          "No": 0,
                           "No internet service": 2,
                           "No phone service": 2
    })
    df = df.replace({"Female": 1,
                           "Male": 0  
    })
    for c in df.columns:
        if c == 'monthly_charges' or c== 'total_charges':
            df[c] = df[c]
        elif df[c].any() == 1:
            df[c] = df[c].astype(int)
    return df

def prep_data(df):
    df = replace_missing_values(df)
    df = create_tenure_year(df)
    df = encode_all(df)
    return df

#---------------------#
#       Splitting     #
#---------------------#

def split_data(df):
    train, test = train_test_split(df, random_state = 123, train_size=.8)
    train, validate = train_test_split(train, random_state=123, train_size=.75)
    return train, validate, test


#---------------------#
#       Encoding      #
#---------------------#

def partner_dependents(df):
    '''
    Used to create a new column that encodes and combines the partners and dependents feature into one 
    
    '''
    if df.partner == 0 and df.dependents == 0:
        return 0
    elif df.partner == 1 and df.dependents == 0:
        return 1
    elif df.partner == 0 and df.dependents == 1:
        return 2
    else:
        return 3

def streaming_features(df):
    '''
    Used to create a new column that encodes and combines the streaming movies and streaming tv features into one 
    
    '''
    if df.streaming_tv == 0 and df.streaming_movies == 0:
        return 0
    elif df.streaming_tv == 1 and df.streaming_movies == 0:
        return 1
    elif df.streaming_tv == 0 and df.streaming_movies == 1:
        return 2
    else:
        return 3

def online_features(df):
    '''
    Used to create a new column that encodes and combines the online security and online banking feature into one 
    
    '''
    if df.online_security == 0 and df.online_backup == 0:
        return 0
    elif df.online_security == 1 and df.online_backup == 0:
        return 1
    elif df.online_security == 0 and df.online_backup == 1:
        return 2
    else:
        return 3

def encode_new_columns(df):
    df = online_features(df)
    df = streaming_features(df)
    df = (partner_dependents(df))
    return df