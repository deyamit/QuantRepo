#Decision Trees
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def train(x_train, y_train):
    fulldecisionTree =  DecisionTreeClassifier(random_state =42)

    fulldecisionTree.fit(x_train,y_train)

    return fulldecisionTree

