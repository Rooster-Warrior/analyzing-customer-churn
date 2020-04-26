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

## Requirements

In order to access the report, it is necesary to have a `env.py` file with  variables called:

* `user`
* `host`
* `password`

The py files and the notebook will automatically pull the data, and create a csv file with all the data for easier access. All of the functions used in the report can be found in the relevant py files in this repo. 

## Further Improvements

## Data dictionary: 

`customerID`: Customer ID
`genderCustomer`: gender (female, male)
`SeniorCitizen`: Whether the customer is a senior citizen or not (1, 0)
`PartnerWhether`: the customer has a partner or not (Yes, No)
`Dependents`: Whether the customer has dependents or not (Yes, No)
`tenure`: Number of months the customer has stayed with the company
`PhoneService`: Whether the customer has a phone service or not (Yes, No)
`MultipleLines`: Whether the customer has multiple lines or not (Yes, No, No phone service)
`InternetService`: Customer’s internet service provider (DSL, Fiber optic, No)
`OnlineSecurity`: Whether the customer has online security or not (Yes, No, No internet service)
`OnlineBackup`: Whether the customer has online backup or not (Yes, No, No internet service)
`DeviceProtection`: Whether the customer has device protection or not (Yes, No, No internet service)
`TechSupport`: Whether the customer has tech support or not (Yes, No, No internet service)
`StreamingTV`: Whether the customer has streaming TV or not (Yes, No, No internet service)
`StreamingMovies`: Whether the customer has streaming movies or not (Yes, No, No internet service)
`Contract`: The contract term of the customer (Month-to-month, One year, Two year)
`PaperlessBilling`: Whether the customer has paperless billing or not (Yes, No)
`PaymentMethod`: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
`MonthlyCharges`: The amount charged to the customer monthly
`TotalCharges`: The total amount charged to the customer
`Churn`: Whether the customer churned or not (Yes or No)

