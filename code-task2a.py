import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier


def main():
    life = pd.read_csv('life.csv', encoding='ISO-8859-1', na_values='..', keep_default_na=True)
    life.sort_values('Country Code', inplace=True)
    world = pd.read_csv('world.csv', encoding='ISO-8859-1', na_values='..', keep_default_na=True)
    world.sort_values('Country Code', inplace=True)

    dataset = life.merge(world, on='Country Code')

    features = dataset.drop(['Country', 'Country Code', 'Year', 'Life expectancy at birth (years)', 'Country Name',
                             'Time'], axis=1).astype(float)
    class_label = dataset[['Life expectancy at birth (years)']]

    x_train, x_test, y_train, y_test = train_test_split(features, class_label, train_size=0.70, test_size=0.30,
                                                        random_state=200)

    feature_names = list()
    medians = list()
    for feature in features:
        feature_names.append(feature)
        medians.append(features[feature].median())

    imputer = SimpleImputer(missing_values=np.nan, strategy='median').fit(x_train)
    x_train = imputer.transform(x_train)
    x_test = imputer.transform(x_test)

    means = list()
    variances = list()
    for feature in features:
        means.append(features[feature].mean())
        variances.append(features[feature].var())
    csv_data = pd.DataFrame({'feature': feature_names, 'median': medians, 'mean': means, 'variance': variances})
    csv_data.round(3).to_csv('task2a.csv', index=False)

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print(f'Accuracy of decision tree: {get_decision_tree_acc(x_train, x_test, y_train, y_test):.3f}')
    print(f'Accuracy of k-nn (k=3): {get_knn3_acc(x_train, x_test, y_train, y_test):.3f}')
    print(f'Accuracy of k-nn (k=7): {get_knn7_acc(x_train, x_test, y_train, y_test):.3f}')


def get_decision_tree_acc(x_train, x_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=200, max_depth=3)
    dt.fit(x_train, y_train)
    y_predtree = dt.predict(x_test)
    return accuracy_score(y_test, y_predtree)


def get_knn3_acc(x_train, x_test, y_train, y_test):
    knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn3.fit(x_train, np.ravel(y_train))
    y_pred3 = knn3.predict(x_test)
    return accuracy_score(y_test, y_pred3)


def get_knn7_acc(x_train, x_test, y_train, y_test):
    knn7 = neighbors.KNeighborsClassifier(n_neighbors=7)
    knn7.fit(x_train, np.ravel(y_train))
    y_pred7 = knn7.predict(x_test)
    return accuracy_score(y_test, y_pred7)


if __name__ == "__main__":
    main()
