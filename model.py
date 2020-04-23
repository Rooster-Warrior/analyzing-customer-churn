from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# ---------------------- #
#        Modeling        #
# ---------------------- #

# Decision Tree

def run_clf(X_train, y_train, max_depth):
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=123)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    return clf, y_pred


# KNN

def run_knn(X_train, y_train, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='uniform')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    return knn, y_pred

# Random_forest

def run_rf(X_train, y_train, leaf, max_depth):
    rf = RandomForestClassifier(random_state= 123, min_samples_leaf = leaf, max_depth = max_depth).fit(X_train, y_train)
    y_pred = rf.predict(X_train)
    return rf, y_pred