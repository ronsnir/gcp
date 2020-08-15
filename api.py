# Import libraries
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np


# API definition

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if rfc_over:
        try:
            json_ = request.json
            print(json_)
            
            df_test = pd.DataFrame(json_)

            # create uuid dataframe
            uuid_df = df_test['uuid'].reset_index(drop=True)

            #build function to fill missing values in account columns
            def fill_account_na(col, acc):
                if pd.isnull(acc):
                    if pd.isnull(col):
                        return 0.0
                    else:
                        return col
                else:
                    return col

            # run the function to fill missing account's (when account_worst_status_0_3m is NULL)
            account_no_0_3 = ['account_days_in_dc_12_24m'
            ,'account_days_in_rem_12_24m'
            ,'account_days_in_term_12_24m'
            ,'account_incoming_debt_vs_paid_0_24m'
            ,'account_status'
            ,'account_worst_status_3_6m'
            ,'account_worst_status_6_12m'
            ,'account_worst_status_12_24m']

            is_account_0_3 = 'account_worst_status_0_3m'

            for col in account_no_0_3:
                df_test[col] = df_test[[col, is_account_0_3]].apply(lambda x: fill_account_na(x[col], x[is_account_0_3]), axis=1)

            # run the function to fill missing account's (when account_worst_status_3_6m is NULL)
            account_no_3_6 = ['account_worst_status_6_12m', 'account_worst_status_12_24m']
            is_account_3_6 = 'account_worst_status_3_6m'
            for col in account_no_3_6:
                df_test[col] = df_test[[col, is_account_3_6]].apply(lambda x: fill_account_na(x[col], x[is_account_3_6]), axis=1)

            # run the function to fill missing account's (when account_worst_status_6_12m is NULL)
            account_no_6_12 = ['account_worst_status_12_24m']
            is_account_6_12 = 'account_worst_status_6_12m'
            for col in account_no_6_12:
                df_test[col] = df_test[[col, is_account_6_12]].apply(lambda x: fill_account_na(x[col], x[is_account_6_12]), axis=1)

            #build function to replace missing values with 0
            def replace_na_zero(col):
                if pd.isnull(col):
                    return 0.0
                else:
                    return col

            #replace all NULL's in account_worst with o.
            columns_replace_na_zero = ['account_worst_status_0_3m'
            ,'account_worst_status_3_6m'
            ,'account_worst_status_6_12m'
            ,'account_worst_status_12_24m']

            for col in columns_replace_na_zero:
                df_test[col] = df_test[col].apply(lambda x: replace_na_zero(x))

            #replace NULL's with 0 in worst_status_active_inv
            df_test['worst_status_active_inv'] = df_test['worst_status_active_inv'].apply(lambda x: replace_na_zero(x))

            #replace NULL's with 0 in account_incoming_debt_vs_paid_0_24m
            df_test['account_incoming_debt_vs_paid_0_24m'] = df_test['account_incoming_debt_vs_paid_0_24m'].apply(lambda x: replace_na_zero(x))

            def age_split(age):
                if (age>=18) & (age<=24):
                    return '18-24'
                elif (age>=25) & (age<=34):
                    return '25-34'
                elif (age>=35) & (age<=44):
                    return '35-44'
                elif (age>=45) & (age<=54):
                    return '45-54'
                elif (age>=55) & (age<=64):
                    return '55-64'
                elif (age>=65) & (age<=74):
                    return '65-74'
                elif (age>=75) & (age<=84):
                    return '75-84'
                elif (age>=85):
                    return '85<'
                else:
                    return avg_age

            #run age_split function
            df_test['age_group'] = df_test['age'].apply(lambda x: age_split(x))

            #adding the new 'mean' column from the grouped df_test (avg_span_group) - i merged the two df's
            df_test = df_test.merge(avg_span_group, on=['has_paid','age_group'], how='left')

            #fill NULL's by the new 'mean' column
            df_test['avg_payment_span_0_12m'] = df_test['avg_payment_span_0_12m'].fillna(value=df_test['mean'])

            #fill again NULL's when 'mean' also has NULL's (took the avg of avg_payment_span_0_12m when has_paid is False)
            df_test['avg_payment_span_0_12m'] = df_test['avg_payment_span_0_12m'].fillna(value=avg_span_has_paid_true)

            #drop 'mean' column
            df_test = df_test.drop('mean', axis=1)

            #adding the new 'mean' column from the grouped df_test (avg_span_group_0_3) - i merged the two df's
            df_test = df_test.merge(avg_span_group_0_3, on='has_paid', how='left')

            #fill NULL's by the new 'mean' column
            df_test['avg_payment_span_0_3m'] = df_test['avg_payment_span_0_3m'].fillna(value=df_test['mean'])

            #drop 'mean' column
            df_test = df_test.drop('mean', axis=1)

            #build function ti fill na only when max_paid_inv_0_12m is 0
            def fill_na_condition(col, max_paid):
                if (pd.isnull(col)) & (max_paid==0):
                    return 0
                else:
                    return col

            #run fill_na_condition function
            df_test['num_active_div_by_paid_inv_0_12m'] = df_test[['num_active_div_by_paid_inv_0_12m','max_paid_inv_0_12m']].apply(lambda x: fill_na_condition(x['num_active_div_by_paid_inv_0_12m'], x['max_paid_inv_0_12m']), axis=1)

            #fill NULL's with 0
            df_test['num_active_div_by_paid_inv_0_12m'] = df_test['num_active_div_by_paid_inv_0_12m'].fillna(value=0)

            #drop irrelevant columns
            df_test = df_test.drop(['num_arch_written_off_0_12m', 'num_arch_written_off_12_24m'], axis=1)

            #create dummies list
            dummies_list = ['account_status'
            , 'account_worst_status_0_3m'
            , 'account_worst_status_12_24m'
            , 'account_worst_status_3_6m'
            , 'account_worst_status_6_12m'
            , 'merchant_category'
            , 'has_paid'
            , 'name_in_email'
            , 'status_last_archived_0_24m'
            , 'status_2nd_last_archived_0_24m'
            , 'status_3rd_last_archived_0_24m'
            , 'status_max_archived_0_6_months'
            , 'status_max_archived_0_12_months'
            , 'status_max_archived_0_24_months'
            , 'worst_status_active_inv']

            #create the dummies columns (convert all columns to string)
            dummies = pd.get_dummies(df_test[dummies_list].astype(str), drop_first=True)

            #drop the original categorical variables
            df_test = df_test.drop(dummies_list, axis=1)

            #combine the df_test with the new dummy variables
            df_test = pd.concat([df_test, dummies], axis=1)

            #drop irrelevant columns for the model
            to_drop = ['uuid', 'merchant_group', 'age_group']
            df_test = df_test.drop(to_drop, axis=1)

            #creating sin and cos variables to deal with cyclical feature: time_hours
            df_test['time_hours_sin'] = np.sin(2 * np.pi * df_test['time_hours']/max_hour) # CHECK max_hour
            df_test['time_hours_cos'] = np.cos(2 * np.pi * df_test['time_hours']/max_hour) # CHECK max_hour

            #drop time_hours column
            df_test = df_test.drop('time_hours', axis=1)

            #drop the default column if exists
            df_test_for_model = df_test.drop('default', axis=1, errors='ignore')

            #build function to complete the missing columns in the test data
            def complete_columns(train, test):
                train_col = train.columns.to_list()
                test_col = test.columns.to_list()
                miss_col = []

                for col in train_col:
                    if col in test_col:
                        continue
                    else:
                        miss_col.append(col)
                return miss_col

            #get a list of the missing columns
            miss_col = complete_columns(X,df_test_for_model)

            #run the function to get the missing columns
            newDF = pd.DataFrame(data=np.zeros(len(df_test_for_model)), columns=['test'])
            for n in miss_col:
                n = pd.DataFrame(data=np.zeros(len(df_test_for_model)).reshape(len(df_test_for_model),1), columns=[n])
                newDF = pd.merge(newDF, n, left_index=True, right_index=True)
            newDF = newDF.drop('test', axis=1)

            #add the missing column to the data
            df_test_for_model2 = pd.merge(df_test_for_model, newDF, left_index=True, right_index=True)

            #create the predictions on the test data
            predictions_test = list(rfc_over.predict(df_test_for_model2))

            return jsonify({'predictions_test': str(predictions_test)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    
    else:
        print('Train the model first')
        return ('no model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 12345

    rfc_over = joblib.load('rfc_over.pkl')
    print('rfc_over loaded')

    avg_age = joblib.load('avg_age.pkl')
    print('avg_age loaded')

    avg_span_group = joblib.load('avg_span_group.pkl')
    print('avg_span_group loaded')

    avg_span_has_paid_true = joblib.load('avg_span_has_paid_true.pkl')
    print('avg_span_has_paid_true loaded')

    avg_span_group_0_3 = joblib.load('avg_span_group_0_3.pkl')
    print('avg_span_group_0_3 loaded')

    max_hour = joblib.load('max_hour.pkl')
    print('max_hour loaded')

    X = joblib.load('X.pkl')
    print('X loaded')

    app.run(port=port, debug=True)