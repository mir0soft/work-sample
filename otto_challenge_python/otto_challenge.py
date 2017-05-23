#################################################
####### MIHRAN HAKOBYAN, OTTO CHALLENGE (kaggle)
####### Task of the Challenge: predict the right class of the data
####### RUNTIME of the script approx. 5 min

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop


train = pd.read_csv('/Users/mihran1/Documents/python/otto/train.csv')
test = pd.read_csv('/Users/mihran1/Documents/python/otto/test.csv')

##### SHORT DATA EXPLORATION

print(train.head())
print(train.describe())


X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']
n_classes = len(set(y_train))

le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)

unique, counts = np.unique(y_train, return_counts=True)
occur = dict(zip(unique, counts))

# absolute frequency
target_names = ['Class_{}'.format(i) for i in range(1,10)]
pos = np.arange(len(occur))
plt.bar(pos, occur.values(), align='center')
plt.xticks(pos, target_names)
plt.title('absolute frequency of the classes')

# partly big correlation between individual features
corr = pd.concat([X_train, pd.Series(y_train, name='target')], axis=1).corr()
plt.figure(figsize=(7,7))
sns.heatmap(corr, vmax=1, square=True, xticklabels=False, yticklabels=False, )
plt.title('correlation between features')

# no correlation between feautres and targets (makes sense, since those are classes)
plt.figure(figsize=(12,7))
pos = np.arange(len(corr.target))
plt.bar(pos, corr.target, align='center')
plt.xticks(np.arange(0, len(corr.target), 2.0))
plt.xlabel('feature')
plt.title('correlation between features and target')

####### MACHINE LEARNING

X_train = np.array(X_train)
X_train = preprocessing.scale(X_train)

X_test = np.array(test.drop('id', axis=1))
X_test = preprocessing.scale(X_test)

# Random Forest crossvalidation
def RF_crossval(X, y, n_splits=5, shuffle=True, random_state=42, n_estimators=120):
  cv = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
  print('RANDOM FOREST crossvalidation')
  for i, (tr, val) in enumerate(cv.split(X)):
      RF = RandomForestClassifier(n_estimators=n_estimators)
      RF.fit(X[tr],y[tr])
      prediction = RF.predict(X[val])
      prediction_proba = RF.predict_proba(X[val])
      score_log_loss = log_loss(y[val], prediction_proba)
      score_accuracy = accuracy_score(y[val], prediction)
      print('fold ' +  str(i) + ':')
      print('log-loss: ' + str(score_log_loss))
      print('Accuray: ' + str(score_accuracy))

# approx. 81% accuracy, 0.59 log-loss
RF_crossval(X_train, y_train)

# XGBoost crossvalidation
def XGB_crossval(X, y,  n_splits=5, shuffle=True, random_state=42, eta=0.1, max_depth=3, n_estimators=100):
  cv = model_selection.KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
  print('XGBOOST')
  for i, (tr, val) in enumerate(cv.split(X)):
      XGB = xgb.XGBClassifier()
      XGB.fit(X[tr],y[tr])
      prediction_proba = XGB.predict_proba(X[val])
      prediction = XGB.predict(X[val])
      score_log_loss = log_loss(y[val], prediction_proba)
      score_accuracy = accuracy_score(y[val], prediction)
      print('fold ' +  str(i) + ':')
      print('log-loss: ' + str(score_log_loss))
      print('Accuray: ' + str(score_accuracy))

# approx. 77% accuracy, 0.65 log-loss
XGB_crossval(X_train, y_train)

####### NEURAL NET

def NN(X,y,epochs=10, validation_split=0.1):

  onehot_label = to_categorical(y, num_classes= 9)

  model = Sequential([
      Dense(100, activation='relu', input_dim=93),
      Dropout(rate=0.5),
      #Dense(50, activation='relu'),
      #Dropout(rate=0.5),
      Dense(9, activation='softmax'),
  ])

  model.compile(optimizer='Adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  print('NEURAL NET')
  model.fit(X, onehot_label, epochs=epochs, validation_split=validation_split)

# Bad performance with NN (approx 18% ), despite several configurations (number layers, number neurons). Overfitting is a big issue here
NN(X_train, y_train)

#### BAGGING
# Very good performance with approx. 0.49 log-loss

def bagging(X, y, test_size=0.2, random_state=13):
  X, X_val, y, y_val = model_selection.train_test_split(X,y, test_size=test_size, random_state=random_state)
  trind = list(range(len(y)))
  pred = np.empty([X_val.shape[0], n_classes])
  B = 10 # number of bagging rounds, should be much higher (runtime increases)
  tmpL = train.shape[0]
  print('BAGGING')

  for i in range(B):
      print('Bagging progress: ' +  str(i))
      # bootstrapped sample
      tmpS1 = np.random.choice(trind, len(trind), replace=True)
      # bootstrapped rest sample
      tmpS2 = list(set(trind) - set(tmpS1))

      tmpX2 = X[tmpS2] # bootstrapped rest train features
      tmpY2 = y[tmpS2] #  bootstrapped rest train labels

      # train RF on bootstrapped rest train
      rf = RandomForestClassifier()
      rf.fit(tmpX2, tmpY2)

      tmpX1 = X[tmpS1] # bootstrapped train
      tmpY1 = y[tmpS1] # bootstrapped train

      # generate new features with RF on bootstrapped train
      tmpX2 = rf.predict_proba(tmpX1) # new training feautres from boostrapped  train
      tmpX3 = rf.predict_proba(X_val) # new validation feature from X_val

      XGB = xgb.XGBClassifier(max_depth=11, learning_rate=0.1, min_child_weight=10, objective = "multi:softprob", nthread=7)

      # train XGB on new and old train features 
      XGB.fit(np.concatenate((tmpX1, tmpX2), axis=1), tmpY1)

      # predict on new and old val features
      pred0 = XGB.predict_proba(np.concatenate((X_val, tmpX3), axis=1))
      pred = pred + pred0

  pred = pred/float(B)
  print('logloss: ' + str(log_loss(y_val, pred)))

# No call due to the long runtime
'''bagging(X_train, y_train)'''

plt.show()

# SUBMISSION
# We will take Random Forest, retrain on the whole set and make predictions

RF = RandomForestClassifier(n_estimators=120)
RF.fit(X_train, y_train)
prediction_proba = RF.predict_proba(X_test)
sub = pd.DataFrame(prediction_proba)
sub.insert(loc=0, column='id', value=test['id'])
sub.columns = ['id'] + target_names
sub.to_csv("submission_rf.csv", index=False)
