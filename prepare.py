import pandas as pd 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import acquire
import model

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


def create_target_dataframes(telco):
    train, validate, test = split_data(telco)
    y_train = train["churn"]
    y_validate = validate["churn"]
    y_test = test["churn"]
    
    return y_train, y_validate, y_test

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
    features_to_drop = ["total_charges", "internet_service_type_id","internet_service_type","contract_type", "payment_type","tech_support", "device_protection", "phone_service","tenure_years", "churn", "streaming_tv", "streaming_movies", "partner", "dependents", "online_security", "online_backup",  "is_churn", "customer_id", "gender", "contract_type_id", "payment_type_id"]

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



# ------------------ #
#      CSV File      # 
# ------------------ # 

def ohe_csv(col, df):
    ''' 
    Function to fit and transform a OneHotEncoder to encode columns and replace the new values in our train, validate, and test df
    '''
    # Creates and fits that encoder
    encoder = OneHotEncoder().fit(df[[col]])
    
    # Transforms and replaced the new columns back to the dataframes
    m = encoder.transform(df[[col]]).todense()
    df = pd.concat([df, pd.DataFrame(m, columns= col + "_" + encoder.categories_[0], index=df.index)], axis = 1)
    return df
    
def return_values_csv(scaler, df):
    '''
    Helper function used to updated the scaled arrays and transform them into usable dataframes
    '''
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns.values).set_index([df.index.values])
    return scaler, df_scaled

# Linear scaler
def min_max_scaler_csv(df):
    '''
    Helper function that scales that data. Returns scaler, as well as the scaled dataframes
    '''
    scaler = MinMaxScaler().fit(df)
    scaler, df_scaled = return_values_csv(scaler, df)
    return scaler, df_scaled

def replace_scaled_values_csv(df, df_scaled):
    ''' 
    Helper function used to replaced the scaled values back into the dataframes
    '''
    for i in range(0,2):
        feature = df_scaled.columns[i]
        df[feature] = df_scaled[feature]
    return df

def full_prep_for_csv(df):
    # We prepare the data by adding missing values
    df = prep_data(df)
    # We encode new columns
    df = encode_new_columns(df)
    # We split the data
    
    # Encode Features
    df = ohe_csv("payment_type", df)
    df = ohe_csv("contract_type", df)
    df = ohe_csv("internet_service_type", df)

    # Drop features
    features_to_drop = ["total_charges","internet_service_type_id","internet_service_type","contract_type", "payment_type","tech_support", "device_protection", "phone_service","tenure_years", "churn", "streaming_tv", "streaming_movies", "partner", "dependents", "online_security", "online_backup",  "is_churn", "customer_id", "gender", "contract_type_id", "payment_type_id"]
    df = df.drop(columns=features_to_drop)

    # select features that we need to scale
    features_to_scale = ["monthly_charges", "tenure"]

    df_scale = df[features_to_scale]

    # scale features
    _, df_scaled = min_max_scaler_csv(df_scale)

    # replaced features with scaled features
    df_scaled = replace_scaled_values(df, df_scaled)
   
    return df_scaled

# ------------------------ #
#   Generate CSV Report    #
# ------------------------ #

def create_csv_df():
    # Read data
    telco = acquire.read_telco_data()
    # Create splits
    X_train, _, _ = full_prep_for_modeling_encoded(telco)
    y_train, _, _ = create_target_dataframes(telco)
    telco_scaled = full_prep_for_csv(telco)
    
    # fit the model using train df
    rf, _ = model.run_rf(X_train, y_train, 1, 8)
    
    # Set a manual threshold for probability of churn
    threshold = 0.4

    predicted_proba = rf.predict_proba(telco_scaled)
    predicted = (predicted_proba [:,1] >= threshold).astype('int')
    
    csv_df = pd.DataFrame({"customer_id": telco.customer_id, "probability_of_churn":predicted_proba[:,1], "prediction_of_churn": predicted})
    
    return csv_df

def create_csv_report():
    '''
    Function creates the report as a DF and then creates a csv file in the current directory
    '''
    csv_df = create_csv_df()
    csv_df.to_csv("telco_customer_churn_predictions.csv")

def check_for_csv_report(file_name):
    '''
    Checks if there is a csv file with the matching name in the directory. If there isn't 
    it will create a new csv using the env file in the directory. 
    '''
    if os.path.exists(file_name) == False:
        create_csv_report()