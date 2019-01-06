#import some necessary librairies

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# %matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points

#Now let's import and put the train and test datasets in  pandas dataframe

train = pd.read_csv('./ad_org_train.csv')
test = pd.read_csv('./ad_org_test.csv')

##display the first five rows of the train dataset.
train.head(5)

##display the first five rows of the test dataset.
test.head(5)

#check the numbers of samples and features
print("The train data size before dropping Id feature is : {} ".format(train.shape))
print("The test data size before dropping Id feature is : {} ".format(test.shape))

#Save the 'Id' column
train_ID = train['vidid']
test_ID = test['vidid']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("vidid", axis = 1, inplace = True)
test.drop("vidid", axis = 1, inplace = True)

#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# Droping the published and duration column

train['year'] = pd.DatetimeIndex(train['published']).year
test['year'] = pd.DatetimeIndex(test['published']).year


train.drop("published", axis = 1, inplace = True)
test.drop("published", axis = 1, inplace = True)

train.drop("duration", axis = 1, inplace = True)
test.drop("duration", axis = 1, inplace = True)




#Concatenate train and test data in same dataframe
ntrain = train.shape[0]
ntest = test.shape[0]
y = train.adview.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['adview'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


all_data['category'] = all_data['category'].apply(str)


# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lbl = LabelEncoder()
all_data['category'] = lbl.fit_transform(list(all_data['category'].values))

# Getting dummy categorical variables
dummy = pd.get_dummies(all_data['category'])
all_data = pd.concat([dummy, all_data], axis=1)
all_data.drop("category", axis = 1, inplace = True)


# Avoiding dummy variable trap
all_data.drop([0], axis=1, inplace = True)



train = all_data[:ntrain]
test = all_data[ntrain:]

# Encoding training and test set
for column in train.columns:
    if train[column].dtype == type(object):
        le = LabelEncoder()
        train[column] = le.fit_transform(train[column])


for column in test.columns:
    if test[column].dtype == type(object):
        le = LabelEncoder()
        test[column] = le.fit_transform(test[column])

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
train = sc_x.fit_transform(train)
test = sc_x.fit_transform(test)
y = sc_y.fit_transform(y)



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.2, random_state = 0)

# Modelling

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb



from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting validation set results
y_pred = regressor.predict(X_test)

# Predicting test set results
y_final = regressor.predict(test)
y_final = sc_y.inverse_transform(y_final)
y_final = np.round(y_final)

sub = pd.DataFrame()
sub['vid_id'] = test_ID
sub['ad_view'] = y_final
sub.to_csv('submission.csv',index=False)
