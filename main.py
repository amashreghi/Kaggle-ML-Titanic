# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import csv
from collections import defaultdict

from sklearn import svm
from sklearn import tree
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

co = 1
d = defaultdict(int)

feat_train = pd.read_csv('titanic/train.csv')
feat_test = pd.read_csv('titanic/test.csv')

#print(feat_train)

# The first rows are names of columns
feature_names = feat_train[0]
label_names = feat_train[0]

def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True


# resolve age missing values
def resolve_missing(data, ageCol, sexCol, fareCol, ticketCol):
    global d
    global co
    for row in data:
        if len(row[ageCol]) == 0:
            row[ageCol] = 0

        if row[sexCol] == 'male':
            row[sexCol] = 0
        else:
            row[sexCol] = 1

        if len(row[fareCol]) == 0:
            row[fareCol] = 0

        if is_int(row[ticketCol]) == False:
            tokens = row[ticketCol].split()
            val = 0
            for i in range(len(tokens)):
                if i == 2:
                    continue
                if is_int(tokens[i]):
                    val = val * 1000 + int(tokens[i])
                else:
                    if not d[tokens[i]]:
                        d[tokens[i]] = co
                        co += 1
                    val = val * 1000 + int(d[tokens[i]])
                row[ticketCol] =  val


def resolve_names(data, nameCol):
    for row in data:
        x = row[nameCol].split(",")
        row[nameCol] = hash(x[0])

nameCol = 3
sexCol = 4
ageCol = 5
sibCol = 6
fareCol = 9
ticketCol = 8
resolve_missing(feat_train, ageCol, sexCol, fareCol, ticketCol)
resolve_missing(feat_test, ageCol - 1, sexCol - 1, fareCol - 1, ticketCol - 1)

resolve_names(feat_train, nameCol)
resolve_names(feat_test, nameCol - 1)

#print(feat_train)

selected_train_features = [2, sexCol, nameCol, fareCol, ticketCol, ageCol, sibCol]

#randnums= np.random.randint(2,892,)

x_train = np.array(feat_train)[1:, selected_train_features].astype(float)
#x_test = np.array(feat_train)[600:, selected_train_features].astype(float)
y_train = np.array(feat_train)[1:, 1].astype(int)
#y_test = np.array(feat_train)[600:, 1].astype(int)

#clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', degree=3, C=2))
#clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, max_iter=100000))
#clf = KNeighborsClassifier(n_neighbors=10, algorithm='brute')
#clf = RandomForestClassifier(max_depth=11, random_state=0);
clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(x_train, y_train)

#simulated_test = clf.predict(x_test)

#print(np.count_nonzero(simulated_test==y_test) / len(y_test))


#final prediction
selected_test_features = [t - 1 for t in selected_train_features]
z = np.array(feat_test)[1:, selected_test_features].astype(float) #pclass age sib
predictions = clf.predict(z)

with open('sumbission.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['PassengerId','Survived'])
    cnt = 0
    for i in predictions:
        spamwriter.writerow([892 + cnt, i])
        cnt+=1
csvfile.close()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
