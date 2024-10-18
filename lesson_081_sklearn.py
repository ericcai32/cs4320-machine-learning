from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

from sklearn.datasets import load_iris

iris_dataset = load_iris()
print("Keys of iris_dataset:", iris_dataset.keys())
print(f"\nPrint names of targets:\n{iris_dataset.target_names}")
print(f"\nPrint the targets:\n{iris_dataset.target}")
print(f"\nShape of the data:\n{iris_dataset['data'].shape}")
print(f"\nFirst 5 rows of feature data:\n{iris_dataset['data'][:5]}")

# Randomize data and do an 80-20 train-test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris_dataset['data'][50:],
                                                    iris_dataset['target'][50:],
                                                    test_size=0.2,
                                                    train_size=0.8,
                                                    random_state=0)

print()
print(f"x_train shape {x_train.shape}")
print(f"x_test shape {x_test.shape}")
print(f"y_train shape {y_train.shape}")
print(f"y_test shape {y_test.shape}")
print(f"y_test is {y_test}")
print("\n------------------------------\n")


# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)   
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

print("K-NEAREST NEIGHBORS")
print(f"Here are our test set predictions for y:\n{y_pred}")
print(f"Here are our actual values for y\n{y_test}")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"Accuracy = {knn.score(x_test, y_test)}")
print("\n------------------------------\n")

# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(solver='newton-cg', max_iter=100)
logr.fit(x_train, y_train)

y_pred = logr.predict(x_test)
print("LOGISTIC REGRESSION")
print(f"Here are our test set predictions for y:\n{y_pred}")
print(f"Here are our actual values for y\n{y_test}")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"Accuracy = {logr.score(x_test, y_test)}")
print("\n------------------------------\n")


# SUPPORT VECTOR MACHINE
from sklearn import svm
svmC = svm.SVC()
svmC.fit(x_train, y_train)
y_pred = svmC.predict(x_test)
print("SUPPORT VECTOR MACHINE")
print(f"Here are our test set predictions for y:\n{y_pred}")
print(f"Here are our actual values for y\n{y_test}")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"Accuracy = {svmC.score(x_test, y_test)}")