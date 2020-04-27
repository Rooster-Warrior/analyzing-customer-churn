import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# ----------------------- #
#       Data Prep         #
# ----------------------- #

def replace_missing_values(df):
    '''
    Helper function used to replaced missing values in the total_charges column with the new calculated total_charges
    '''
    df.total_charges = df.total_charges.replace(" ", df.tenure*df.monthly_charges)
    df.total_charges = df.total_charges.astype(float)
    return df

def create_tenure_year(df):
    '''
    Helper function used to create a new column which looks at tenure in years
    '''
    df["tenure_years"] = df.tenure/12
    df.tenure_years = df.tenure_years.astype(int)
    return df
    
def is_churn(df):
    '''
    Encoding helper function that creates a new column for churn which holds int
    '''
    df['is_churn'] = (df.churn == "Yes")
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
    df = is_churn(df)
    df = encode_all(df)
    return df

#---------------------#
#       Splitting     #
#---------------------#

def split_data(df):
    '''
    Main function to split data into train, validate, and test datasets. Random_state == 123, train_size = .8
    '''
    train, test = train_test_split(df, random_state =123, train_size=.8)
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
    '''
    Main function used to encode new columns, this helps combine columns to help reduced the number of features.
    '''
    df["partner_dependents"] = df.apply(lambda row: partner_dependents(row), axis = 1)

    df["streaming_features"] = df.apply(lambda row: streaming_features(row), axis = 1)

    df["online_features"] = df.apply(lambda row: online_features(row), axis = 1)
    return df

# ---------------------- #
#         Scaling        #
# ---------------------- #


def split_my_df(df):
    train, test = train_test_split(df, train_size=.8, random_state=123)
    return train, test


def return_values(scaler, train, validate, test):
    '''
    Helper function used to updated the scaled arrays and transform them into usable dataframes
    '''
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    validate_scaled = pd.DataFrame(scaler.transform(validate), columns=validate.columns.values).set_index([validate.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train_scaled, validate_scaled, test_scaled

# Linear scaler
def min_max_scaler(train,validate, test):
    '''
    Helper function that scales that data. Returns scaler, as well as the scaled dataframes
    '''
    scaler = MinMaxScaler().fit(test)
    scaler, train_scaled, validate_scaled, test_scaled = return_values(scaler, train, validate, test)
    return scaler, train_scaled, validate_scaled, test_scaled

def drop_features(train, validate, test):
    '''
    Helper function used to define variables to drop, and apply them to our train, valdiate, and test datasets
    '''
    # Drop features
    features_to_drop = ["online_features", "churn", "streaming_tv", "streaming_movies", "partner", "dependents", "online_security", "online_backup",  "is_churn", "customer_id", "gender", "contract_type", "payment_type", "internet_service_type"]

    X_train = train.drop(columns= features_to_drop)
    X_validate = validate.drop(columns=features_to_drop)
    X_test = test.drop(columns=features_to_drop)
    return X_train, X_validate, X_test

def replace_scaled_values(df, df_scaled):
    ''' 
    Helper function used to replaced the scaled values back into the dataframes
    '''
    for i in range(0,2):
        feature = df_scaled.columns[i]
        df[feature] = df_scaled[feature]
    return df


def prep_for_modeling(train, validate, test):
    '''
    Helper function used to scale the data
    '''
    X_train, X_validate, X_test = drop_features(train, validate, test)

    features_to_scale = ["monthly_charges", "tenure"]

    X_train_scale = X_train[features_to_scale]
    X_validate_scale = X_validate[features_to_scale]
    X_test_scale = X_test[features_to_scale]

    _, X_train_scaled,X_validate_scaled, X_test_scaled = min_max_scaler(X_train_scale, X_validate_scale, X_test_scale)

    X_train = replace_scaled_values(X_train, X_train_scaled)
    X_validate = replace_scaled_values(X_validate, X_validate_scaled)
    X_test = replace_scaled_values(X_test, X_test_scaled)

    return X_train, X_validate, X_test

def full_prep_for_modeling(df):
    '''
    Function used to run through all the prep and scaling before modeling. 
    ''' 
    # We prepare the data by adding missing values
    df = prep_data(df)
    # We encode new columns
    df = encode_new_columns(df)
    # We split the data
    train, validate, test = split_data(df)
    # Drop features we ahve determined are not useful
    X_train, X_validate, X_test = drop_features(train, validate, test)
    
    # select features that we need to scale
    features_to_scale = ["monthly_charges", "total_charges", "tenure", "tenure_years"]

    X_train_scale = X_train[features_to_scale]
    X_validate_scale = X_validate[features_to_scale]
    X_test_scale = X_test[features_to_scale]

    # scale features
    _, X_train_scaled,X_validate_scaled, X_test_scaled = min_max_scaler(X_train_scale, X_validate_scale, X_test_scale)

    # replaced features with scaled features
    X_train = replace_scaled_values(X_train, X_train_scaled)
    X_validate = replace_scaled_values(X_validate, X_validate_scaled)
    X_test = replace_scaled_values(X_test, X_test_scaled)

    return X_train, X_validate, X_test

def ohe(col, X_train, X_validate, X_test):
    ''' 
    Function to fit and transform a OneHotEncoder to encode columns and replace the new values in our train, validate, and test df
    '''
    # Creates and fits that encoder
    encoder = OneHotEncoder().fit(X_train[[col]])
    
    # Transforms and replaced the new columns back to the dataframes
    m = encoder.transform(X_train[[col]]).todense()
    X_train = pd.concat([X_train, pd.DataFrame(m, columns= col + encoder.categories_[0], index=X_train.index)], axis = 1)
    
    m = encoder. transform(X_validate[[col]]).todense()
    X_validate = pd.concat([X_validate, pd.DataFrame(m, columns=col + encoder.categories_[0], index = X_validate.index)], axis = 1)
    
    m = encoder. transform(X_test[[col]]).todense()
    X_test = pd.concat([X_test, pd.DataFrame(m, columns=col + encoder.categories_[0], index = X_test.index)], axis = 1)
    
    return X_train, X_validate, X_test



def full_prep_for_modeling_encoded(df):
    '''
    Function used to prepare the data before modeling. This function has the additional functionality of fitting and transforming columns using a OneHotEncoder
    '''
    # We prepare the data by adding missing values
    df = prep_data(df)
    # We encode new columns
    df = encode_new_columns(df)
    # We split the data
    train, validate, test = split_data(df)

    # Encode Features
    train, validate, test = ohe("payment_type", train, validate, test)
    train, validate, test = ohe("contract_type", train, validate, test)
    train, validate, test = ohe("internet_service_type", train, validate, test)
    
    


    # Drop features
    features_to_drop = ["internet_service_type_id","internet_service_type","contract_type", "payment_type","tech_support", "device_protection", "phone_service","tenure_years", "churn", "streaming_tv", "streaming_movies", "partner", "dependents", "online_security", "online_backup",  "is_churn", "customer_id", "gender", "contract_type_id", "payment_type_id"]

    X_train = train.drop(columns= features_to_drop)
    X_validate = validate.drop(columns=features_to_drop)
    X_test = test.drop(columns=features_to_drop)
       
    
    # select features that we need to scale
    features_to_scale = ["monthly_charges", "tenure"]

    X_train_scale = X_train[features_to_scale]
    X_validate_scale = X_validate[features_to_scale]
    X_test_scale = X_test[features_to_scale]

    # scale features
    _, X_train_scaled,X_validate_scaled, X_test_scaled = min_max_scaler(X_train_scale, X_validate_scale, X_test_scale)

    # replaced features with scaled features
    X_train = replace_scaled_values(X_train, X_train_scaled)
    X_validate = replace_scaled_values(X_validate, X_validate_scaled)
    X_test = replace_scaled_values(X_test, X_test_scaled)

    
    return X_train, X_validate, X_test
