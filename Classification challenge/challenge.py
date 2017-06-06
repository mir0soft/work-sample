#################################################
####### MIHRAN HAKOBYAN
####### Task of the Challenge: predict the right class of the data
####### RUNTIME of the script approx. 5 min

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing, model_selection, linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import seaborn as sns


names = ['feature{}'.format(x) for x in range(295)]
names.append('label')

# we will use only 30% of the data for shorter runtime
data = pd.read_csv('/Users/mihran1/Documents/python/sample.csv', names=names)
data = data.sample(frac=0.3, random_state=11)


###### DATA EXPLORATION ########

# most of the features are sparse; the nonzero values are mostly 1, some features have a much greater value
print(data.head())
print(data.describe())

# high correlation between few features, mostly approx. zero
corr = data.corr()
plt.figure()
sns.heatmap(corr, vmax=1, square=True, xticklabels=False, yticklabels=False)
plt.title('correlation between Features')

# distribution of the target variable; class C occurs much more often then the other classes
plt.figure()
counts = data['label'].value_counts()
y_pos = np.arange(len(counts.index))
plt.bar(y_pos, counts.values)
plt.xticks(y_pos, counts.index)
plt.title('absolute frequency of the classes')

dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
data['label'] = data['label'].apply(lambda x: dic[x])


######## DATA PREPARATION AND FEATURE ENGINEERING  ######

X = data.drop('label', axis=1)
y = data['label']

# remove non changing variables
nuniques = X.apply(lambda x:x.nunique())
no_variation = nuniques[nuniques==1].index
X = X.drop(no_variation, axis=1)
print(len(no_variation), 'variables do not change (removed).')

# add new features
X['mean'] = X.apply(lambda x: np.mean(x), axis=1)
X['std'] = X.apply(lambda x: np.std(x), axis=1)
X['nnonzeros'] = X.apply(lambda x: len(x[x!=0]), axis=1)
X['nzeros'] = X.apply(lambda x: len(x[x==0]), axis=1)
X['max'] = X.apply(lambda x: x.max(), axis=1)
X['min'] = X.apply(lambda x: x.min(), axis=1)

# convert and scale (to zero mean and standard deviation of 1)
X = np.array(X)
y = np.array(y)
X = preprocessing.scale(X)

# separate in training set and validation set
# no crossvalidation here, so we can try Model Stacking

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,  test_size=0.3)

###### SINGLE RANDOM FOREST ###########
# first let´s try only Random Forst:

RF = RandomForestClassifier()
RF.fit(X_train, y_train)
predictions = RF.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, predictions))

########## VOTING CLASSIFIER ##################
# now we want to try a Voting Classifier consisting of three classifiers
# I did not tune the parameters, since this is a demo; GridSearch would be wise

RF = RandomForestClassifier()
LR = linear_model.LogisticRegression()
XGB = xgb.XGBClassifier()

voting = VotingClassifier([('RF', RF), ('LR', LR), ('XGB', XGB)])
voting.fit(X_train, y_train)
predictions = voting.predict(X_test)
print('Voting Classifier Accuracy:', accuracy_score(y_test, predictions))


######## STACKING MODEL ###########
ntrain = X_train.shape[0]
ntest = X_test.shape[0]
seed = 0
nFolds = 5
kf = model_selection.KFold(n_splits=nFolds, shuffle=True, random_state=seed)

# This function computes meta-features for a given model
def get_meta(clf, x_train, y_train, x_test):
   meta_train = np.zeros((ntrain,))
   meta_test = np.zeros((ntest,))
   meta_test_skf = np.empty((nFolds, ntest))

   for i, (train_index, test_index) in enumerate(kf.split(x_train)):
       x_tr = x_train[train_index]
       y_tr = y_train[train_index]
       x_te = x_train[test_index]

       clf.fit(x_tr, y_tr)

       meta_train[test_index] = clf.predict(x_te)
       meta_test_skf[i, :] = clf.predict(x_test)

   meta_test[:] = meta_test_skf.mean(axis=0)
   meta_train = meta_train.reshape(-1, 1)
   meta_test = meta_test.reshape(-1, 1)
   return meta_train, meta_test

ET_meta_train, ET_meta_test = get_meta(ExtraTreesClassifier(), X_train, y_train, X_test) # Extra Trees

RF_meta_train, RF_meta_test = get_meta(RandomForestClassifier(),X_train, y_train, X_test) # Random Forest

ADA_meta_train, ADA_meta_test = get_meta(AdaBoostClassifier(), X_train, y_train, X_test) # AdaBoost

LR_meta_train, LR_meta_test = get_meta(linear_model.LogisticRegression(),X_train, y_train, X_test) # Logistic Regression

KNN_meta_train, KNN_meta_test = get_meta(KNeighborsClassifier(), X_train, y_train, X_test) # k Nearest Neighbors

# bind computed meta-features
x_train_stack = np.concatenate((ET_meta_train, RF_meta_train, ADA_meta_train, LR_meta_train, KNN_meta_train), axis=1)
x_test_stack = np.concatenate((ET_meta_test, RF_meta_test, ADA_meta_test, LR_meta_test, KNN_meta_test), axis=1)

# we will use the meta-features to train the strong XGBoost and make the predictions
XGB = xgb.XGBClassifier().fit(x_train_stack, y_train)
predictions = XGB.predict(x_test_stack)
print('Model Stacking accuracy:', accuracy_score(y_test, predictions))


# lets have a look at the feature importances of XGBoost
meta_feature_names = ['Extra Trees','RandomForest', 'AdaBoost', 'Log Reg', 'KNN']

feature_importances = pd.Series(XGB.feature_importances_, meta_feature_names)
feature_importances.sort_values(inplace=True)

plt.figure(figsize=(12,7))
feature_importances.plot(kind='barh', figsize=(12,7), title='Feature importances')

# what about the correlation between the meta-features, the less the better
base_predictions_train = pd.DataFrame( {'RandomForest': RF_meta_train.ravel(),
     'ExtraTrees': ET_meta_train.ravel(),
     'AdaBoost': ADA_meta_train.ravel(),
      'KNN': KNN_meta_train.ravel(),
      'Log Reg': LR_meta_train.ravel()
    })

plt.figure(figsize=(10,7))
sns.heatmap(base_predictions_train.astype(float).corr())
plt.title('correlation between meta-features')
plt.show()

###### CONCLUSION ######
# We can see an accuracy improvement compared to the simple Random Forest due to the Voting and Stacking techniques. Next we should try a Neural Net.
