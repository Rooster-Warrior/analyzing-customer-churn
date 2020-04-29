# Analyzing Customer Churn Project

## Objectives:  

### 1. To understand why telcom customers are churning

In order to answer this question - we looked to answer these more concrete questions first: 

1. Are there clear groupings where a customer is more likely to churn? What if you consider contract type? Is there a tenure that month-to-month customers are most likely to churn? 1-year contract customers? 2-year customers? Do you have any thoughts on what could be going on? 

1. Plot the rate of churn on a line chart where x is the tenure and y is the rate of churn (customers churned/total customers).

1. Are there features that indicate a higher propensity to churn? like type of internet service, type of phone service, online security and backup, senior citizens, paying more than x% of customers with the same services, etc.?

1. Is there a price threshold for specific services where the likelihood of churn increases once price for those services goes past that point? If so, what is that point for what service(s)?

1. If we looked at churn rate for month-to-month customers after the 12th month and that of 1-year contract customers after the 12th month, are those rates comparable?

### 2. To use classification models to predict customers that would be at risk of churning, and offering insights to help the company prevent further churning.

## Executive Summary
The data science team was asked by senior management to analyze customer churn.   The overall goal of this project was to examine if identifying churned customers is a potential revenue source, and if it is what is the most effective way to predict which customers are likely to churn.  

The data was supplied from subscription data acquired by customer sign ups from 'telco_churn' database on Codeup SQL Server.  Using 7043 customer entries, we built reproducible functions to acquire, prepare, explore and model customer data.  The results can be found in "telco_customer_churn_predictions.csv" on the company google classroom server.

An exploration of features important to predicting churn were two continuous variables: tenure and monthly charges.   The categorical features that affected customer churn are contract type, internet service type, and online features (a derived column of whether the customer used online back up and online payments).   Dependents and partner were two features that had a high variance in churned compared to retained customers, but they did not factor greatly in the predictive models.  Interestingly there is a clear distinction in pricing differences between retained and churned customers with an $20 a month per customer difference on average.  

Using Random Forest, an advanced decision tree modeling algorithm, we were able to predict customer churn with an 80% success rate.  Due to the high cost of acquisition of new customers we recommend focussing on reducing the occurrences of losing a customer who we have not labelled as a potential churn.  

There is a clear business case for focussing on reducing customer churn.  We recommend focussing on month-to-month customers and trying to grow them into contracted customers.  This can be achieved in two different ways: education through marketing, or redefinition of contract types.  The education would focus on highlighting benefits of company products and services.  Consequently, converting all month-to-month contracts into yearly contracts with a cancel anytime option may provide a psychological bump.  Since most customers churn within the first 6 months, a value add of premium features for the first 6 months may also have the desired results.  

Recommendations:

- Follow up calls and marketing materials sent to newly acquired month-to-month customers
- Change all month-to-month customers to contract with no cancellation fee
- Offer value add services for new month-to-month customers to reduce cancellations
- Push internet services on phone only customers
- Push fiber optic service on DSL customers
- It may be worth seeing if we can improve the accuracy and recall of our model by using only month-to-month customers in data set. 
- If all else fails, offer discounts and credits to customers who remain with company for a year

## Requirements

In order to access the report, it is necesary to have a `env.py` file with  variables called:

* `user`
* `host`
* `password`

The py files and the notebook will automatically pull the data, and create a csv file with all the data for easier access. All of the functions used in the report can be found in the relevant py files in this repo. 

### To create a new CSV report with new data:

* Use the `create_csv_report_new_data` function in the model.py file. The function will use the credentials in the env.py file to pull the new data from the SQL database, and create a local CSV file. 
    * The function will then prepare the data for modeling by filling missing values, dropping unwanted columns, scaling the numerical features, and encoding the categorical features. The model will be run, using the same hyperparameters as used on the report, and store the desired featuers in a CSV file.

## Further Improvements
* Additional information, in the form of a binary column, on whether the customer churn because they were relocating, or because they had a negative product or customer experience. If we can remove these outliers, we can then make the model far more accurate at predicting customers that will churn based on price. 

## Data dictionary: 

* `customerID`: Customer ID
* `genderCustomer`: gender (female, male)
* `SeniorCitizen`: Whether the customer is a senior citizen or not (1, 0)
* `PartnerWhether`: the customer has a partner or not (Yes, No)
* `Dependents`: Whether the customer has dependents or not (Yes, No)
* `tenure`: Number of months the customer has stayed with the company
* `PhoneService`: Whether the customer has a phone service or not (Yes, No)
* `MultipleLines`: Whether the customer has multiple lines or not (Yes, No, No phone service)
* `InternetService`: Customer’s internet service provider (DSL, Fiber optic, No)
* `OnlineSecurity`: Whether the customer has online security or not (Yes, No, No internet service)
* `OnlineBackup`: Whether the customer has online backup or not (Yes, No, No internet service)
* `DeviceProtection`: Whether the customer has device protection or not (Yes, No, No internet service)
* `TechSupport`: Whether the customer has tech support or not (Yes, No, No internet service)
* `StreamingTV`: Whether the customer has streaming TV or not (Yes, No, No internet service)
* `StreamingMovies`: Whether the customer has streaming movies or not (Yes, No, No internet service)
* `Contract`: The contract term of the customer (Month-to-month, One year, Two year)
* `PaperlessBilling`: Whether the customer has paperless billing or not (Yes, No)
* `PaymentMethod`: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
* `MonthlyCharges`: The amount charged to the customer monthly
* `TotalCharges`: The total amount charged to the customer
* `Churn`: Whether the customer churned or not (Yes or No)

