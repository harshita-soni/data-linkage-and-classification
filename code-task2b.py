import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from pprint import pprint as pp


def main():
    life = pd.read_csv('life.csv', encoding='ISO-8859-1', na_values='..', keep_default_na=True)
    life.sort_values('Country Code', inplace=True)
    world = pd.read_csv('world.csv', encoding='ISO-8859-1', na_values='..', keep_default_na=True)
    world.sort_values('Country Code', inplace=True)

    dataset = life.merge(world, on='Country Code')

    features = dataset.drop(['Country', 'Country Code', 'Year', 'Life expectancy at birth (years)', 'Country Name',
                             'Time'], axis=1).astype(float)
    class_label = dataset[['Life expectancy at birth (years)']]

    acc_fe = do_feature_engineering(features, class_label)
    acc_pca = do_pca(features, class_label)
    acc_ff = do_first_four_features(features, class_label)

    print(f'Accuracy of feature engineering: {acc_fe:.3f}')
    print(f'Accuracy of PCA: {acc_pca:.3f}')
    print(f'Accuracy of first four features: {acc_ff:.3f}')


def do_feature_engineering(features, class_label):
    x_train, x_test, y_train, y_test = train_test_split(features, class_label, train_size=0.70, test_size=0.30,
                                                        random_state=200, stratify=class_label)
    imputer = SimpleImputer(missing_values=np.nan, strategy='median').fit(x_train)
    column_names = list(x_train.columns)
    comb = list(combinations(column_names, 2))
    for col1, col2 in comb:
        column_names.append(col1+' * '+col2)

    x_train = imputer.transform(x_train)
    x_test = imputer.transform(x_test)

    poly = PolynomialFeatures(interaction_only=True, include_bias=False).fit(x_train)
    x_train = poly.transform(x_train)
    x_test = poly.transform(x_test)

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = pd.DataFrame(x_train, columns=column_names)
    x_test = pd.DataFrame(x_test, columns=column_names)

    sum_of_squares = list()
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=200).fit(x_train.iloc[:, :20])
        sum_of_squares.append(kmeans.inertia_)

    plt.plot(range(1, 11), sum_of_squares)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of Squares (within cluster)')
    plt.show()
    plt.savefig('task2bgraph1.png', dpi=400, bbox_inches='tight')

    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=200).fit(x_train.iloc[:, :20])
    x_train['Cluster label'] = kmeans.labels_
    x_test['Cluster label'] = kmeans.predict(x_test.iloc[:, :20])

    print("x_train after feature engineering (head):")
    pp(x_train.iloc[:5, :])

    fs = SelectKBest(score_func=mutual_info_classif, k=4)
    fs.fit(x_train, np.ravel(y_train))
    x_train = fs.transform(x_train)
    x_test = fs.transform(x_test)

    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)

    print("\nFour selected features after feature engineering (head):")
    pp(x_train.iloc[:5, :])

    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, np.ravel(y_train))
    y_kmeans = knn.predict(x_test)
    return accuracy_score(y_test, y_kmeans)


def do_pca(features, class_label):
    x_train, x_test, y_train, y_test = train_test_split(features, class_label, train_size=0.70, test_size=0.30,
                                                        random_state=200, stratify=class_label)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean').fit(x_train)
    x_train = imputer.transform(x_train)
    x_test = imputer.transform(x_test)

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    pca = PCA(n_components=4)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)

    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)

    print("\nx_train after PCA (head):")
    pp(x_train.iloc[:5, :])

    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, np.ravel(y_train))
    y_pca = knn.predict(x_test)
    return accuracy_score(y_test, y_pca)


def do_first_four_features(features, class_label):
    x_train, x_test, y_train, y_test = train_test_split(features.iloc[:, :4], class_label, train_size=0.70,
                                                        test_size=0.30, random_state=200)
    column_names = x_train.columns

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean').fit(x_train)
    x_train = imputer.transform(x_train)
    x_test = imputer.transform(x_test)

    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = pd.DataFrame(x_train, columns=column_names)
    x_test = pd.DataFrame(x_test, columns=column_names)

    print("\nFirst four features (head):")
    pp(x_train.iloc[:5, :])
    print('\n')

    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, np.ravel(y_train))
    y_first_four = knn.predict(x_test)
    return accuracy_score(y_test, y_first_four)


if __name__ == "__main__":
    main()
