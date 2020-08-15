# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import joblib


#get the data
df = pd.read_csv('/home/ubuntu/aws/dataset.csv', sep=';')

#split the data for train (exclude test)
df_train = df[df['default'].notnull()]


### NULLS ###

#### account_worst_status ####

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
    df_train[col] = df_train[[col, is_account_0_3]].apply(lambda x: fill_account_na(x[col], x[is_account_0_3]), axis=1)

# run the function to fill missing account's (when account_worst_status_3_6m is NULL)
account_no_3_6 = ['account_worst_status_6_12m', 'account_worst_status_12_24m']
is_account_3_6 = 'account_worst_status_3_6m'
for col in account_no_3_6:
    df_train[col] = df_train[[col, is_account_3_6]].apply(lambda x: fill_account_na(x[col], x[is_account_3_6]), axis=1)

# run the function to fill missing account's (when account_worst_status_6_12m is NULL)
account_no_6_12 = ['account_worst_status_12_24m']
is_account_6_12 = 'account_worst_status_6_12m'
for col in account_no_6_12:
    df_train[col] = df_train[[col, is_account_6_12]].apply(lambda x: fill_account_na(x[col], x[is_account_6_12]), axis=1)

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
    df_train[col] = df_train[col].apply(lambda x: replace_na_zero(x))

#### END account_worst_status ####


#### worst_status_active_inv ####

#replace NULL's with 0 in worst_status_active_inv
df_train['worst_status_active_inv'] = df_train['worst_status_active_inv'].apply(lambda x: replace_na_zero(x))

#### END worst_status_active_inv ####


#### account_incoming_debt_vs_paid_0_24m ####

#replace NULL's with 0 in worst_status_active_inv
df_train['account_incoming_debt_vs_paid_0_24m'] = df_train['account_incoming_debt_vs_paid_0_24m'].apply(lambda x: replace_na_zero(x))

#### END account_incoming_debt_vs_paid_0_24m ####


############# span 0_12 ############

#create avg_age
avg_age = df_train['age'].mean()

#build function to split age into groups
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
df_train['age_group'] = df_train['age'].apply(lambda x: age_split(x))

#create new df_train where I group by has_paid AND age_group to get the avg avg_payment_span_0_12m
avg_span_group = df_train.groupby(['has_paid', 'age_group']).mean()['avg_payment_span_0_12m'].reset_index(name='mean')

#adding the new 'mean' column from the grouped df_train (avg_span_group) - i merged the two df's
df_train = df_train.merge(avg_span_group, on=['has_paid','age_group'], how='left')

#fill NULL's by the new 'mean' column
df_train['avg_payment_span_0_12m'] = df_train['avg_payment_span_0_12m'].fillna(value=df_train['mean'])

#create avg_span_has_paid_true
avg_span_has_paid_true = df_train[df_train['has_paid']==False]['avg_payment_span_0_12m'].mean()

#fill again NULL's when 'mean' also has NULL's (took the avg of avg_payment_span_0_12m when has_paid is False)
df_train['avg_payment_span_0_12m'] = df_train['avg_payment_span_0_12m'].fillna(value=avg_span_has_paid_true)

#drop 'mean' column
df_train = df_train.drop('mean', axis=1)

############# END OF span 0_12 ############


############# span 0_3 ############

#create new df_train where I group by has_paid to get the avg of avg_payment_span_0_3m
avg_span_group_0_3 = df_train.groupby('has_paid').mean()['avg_payment_span_0_3m'].reset_index(name='mean')

#adding the new 'mean' column from the grouped df_train (avg_span_group_0_3) - i merged the two df's
df_train = df_train.merge(avg_span_group_0_3, on='has_paid', how='left')

#fill NULL's by the new 'mean' column
df_train['avg_payment_span_0_3m'] = df_train['avg_payment_span_0_3m'].fillna(value=df_train['mean'])

#drop 'mean' column
df_train = df_train.drop('mean', axis=1)

############# END OF span 0_3 ############


############# THE RATIO ############

#build function ti fill na only when max_paid_inv_0_12m is 0
def fill_na_condition(col, max_paid):
    if (pd.isnull(col)) & (max_paid==0):
        return 0
    else:
        return col

#run fill_na_condition function
df_train['num_active_div_by_paid_inv_0_12m'] = df_train[['num_active_div_by_paid_inv_0_12m','max_paid_inv_0_12m']].apply(lambda x: fill_na_condition(x['num_active_div_by_paid_inv_0_12m'], x['max_paid_inv_0_12m']), axis=1)

#fill NULL's by the new 'mean' column
df_train['num_active_div_by_paid_inv_0_12m'] = df_train['num_active_div_by_paid_inv_0_12m'].fillna(value=0)

############# END THE RATIO ############


########## num_arch_written_off ############

#drop irrelevant columns
df_train = df_train.drop(['num_arch_written_off_0_12m', 'num_arch_written_off_12_24m'], axis=1)

########## END num_arch_written_off ############


######### Dummies #######

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
dummies = pd.get_dummies(df_train[dummies_list].astype(str), drop_first=True)

#drop the original categorical variables
df_train = df_train.drop(dummies_list, axis=1)

#combine the df_train with the new dummy variables
df_train = pd.concat([df_train, dummies], axis=1)

######### END Dummies #######


#drop irrelevant columns for the model
to_drop = ['uuid', 'merchant_group', 'age_group']
df_train = df_train.drop(to_drop, axis=1)

#creating sin and cos variables to deal with cyclical feature: time_hours
max_hour = df_train['time_hours'].max()
df_train['time_hours_sin'] = np.sin(2 * np.pi * df_train['time_hours']/max_hour)
df_train['time_hours_cos'] = np.cos(2 * np.pi * df_train['time_hours']/max_hour)

#drop the original time_hours column
df_train = df_train.drop('time_hours', axis=1)


############ Split The Data To Train-Test ###########

#create datasets for the split
X = df_train.drop('default', axis=1)
y = df_train['default']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=101)

############ END Split The Data To Train-Test ###########


### RANDOM FOREST - unbalance fix (oversampling) ###

#create instance for oversampling
oversample = RandomOverSampler(sampling_strategy='minority')

#create the new datasets withe the new rows
X_over, y_over = oversample.fit_resample(X_train, y_train)

#create Random Forest instance
rfc_over = RandomForestClassifier(n_estimators=25)

#fit the model
rfc_over.fit(X_over, y_over)

### END RANDOM FOREST - unbalance fix (oversampling) ###


### Save the model ###

#save the rfc_over model
joblib.dump(rfc_over, 'rfc_over.pkl')
print('rfc_over dumped!')

#load the model
rfc_over = joblib.load('rfc_over.pkl')

#save entities
joblib.dump(avg_age, 'avg_age.pkl')
print('avg_age dumped!')

joblib.dump(avg_span_group, 'avg_span_group.pkl')
print('avg_span_group dumped!')

joblib.dump(avg_span_has_paid_true, 'avg_span_has_paid_true.pkl')
print('avg_span_has_paid_true dumped!')

joblib.dump(avg_span_group_0_3, 'avg_span_group_0_3.pkl')
print('avg_span_group_0_3 dumped!')

joblib.dump(max_hour, 'max_hour.pkl')
print('max_hour dumped!')

joblib.dump(X, 'X.pkl')
print('X dumped!')
