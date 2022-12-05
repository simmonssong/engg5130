import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_preprocessing import get_data, encode_categorical, normalize_numerical
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



def make_data():
    data = get_data()
    data_preprocessed = pd.concat(
        [
            data[['won']],
            normalize_numerical(),
            encode_categorical()
        ], axis=1)
    # print(data_preprocessed.shape)
    return data_preprocessed


def check_label():
    data_preprocessed = make_data()
    plt.figure(figsize=(6, 4))
    sns.countplot(data=data_preprocessed, x='won')
    plt.title('Number of Labels')
    plt.savefig("results/label_dis.png")

# check_label()


def split_data():
    data_preprocessed = make_data()
    print(data_preprocessed.shape)
    trainset, evlset, testset = np.split(data_preprocessed.sample(frac=1, random_state=42), [
                                         data_preprocessed.shape[0] - 7000 - 7000, data_preprocessed.shape[0] - 7000])

    trainx = trainset.drop(columns=['won'])
    trainy = trainset['won']
    # sample_size = 7000
    # testset = data_preprocessed.sample(sample_size, random_state=414)

    testx = testset.drop(columns=['won'])
    testy = testset['won']

    # evlset = data_preprocessed.sample(sample_size, random_state=200)
    evlx = evlset.drop(columns=['won'])
    evly = evlset['won']

    return trainx, trainy, evlx, evly, testx, testy


trainx, trainy, evlx, evly, testx, testy = split_data()

search_space = {'n_estimators': [int(x) for x in np.linspace(50, 1000, num=20)],
                'max_depth': [int(x) for x in np.linspace(10, 100, num=10)] + [None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy'],
                'bootstrap': [True, False]
                }


def eval(model, model_2, x, y, tx=evlx, ty=evly, is_train=True):
    if is_train:
        model.fit(x, y)

    # model = LogisticRegression(solver='liblinear').fit(trainx, trainy)
    # model = MLPClassifier().fit(trainx, trainy)

    lr_pred = model.predict(tx)
    print("Predict Count", sorted(Counter(lr_pred).items()))
    print('Evaluate score: ', accuracy_score(ty, lr_pred))

    # lr_pred = model.predict(testx)
    # print('Test score: ', accuracy_score(testy, lr_pred))

    print(classification_report(ty, lr_pred))
    print(confusion_matrix(ty, lr_pred))

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x, y)
    # print(sorted(Counter(trainy).items()))
    # print(sorted(Counter(y_resampled).items()))

    if is_train:
        model_2.fit(X_resampled, y_resampled)

    # model = LogisticRegression(solver='liblinear').fit(trainx, trainy)
    # model = MLPClassifier().fit(trainx, trainy)

    lr_pred = model_2.predict(tx)
    print("Predict Count", sorted(Counter(lr_pred).items()))
    print('Evaluate score: ', accuracy_score(ty, lr_pred))

    # lr_pred = model.predict(testx)
    # print('Test score: ', accuracy_score(testy, lr_pred))

    print(classification_report(ty, lr_pred))
    print(confusion_matrix(ty, lr_pred))

    return model, model_2


# model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model = MLPClassifier(hidden_layer_sizes=(64, 8), activation='relu', solver='adam',
                      alpha=0.0001, batch_size=512, learning_rate='constant', learning_rate_init=0.001, max_iter=200)
# model_2 = DecisionTreeClassifier(criterion='entropy', random_state=0)
model_2 = MLPClassifier(hidden_layer_sizes=(1024,), activation='relu', solver='adam',
                      alpha=0.0001, batch_size=512, learning_rate='constant', learning_rate_init=0.001, max_iter=200)

model, model_2 = eval(model, model_2, trainx, trainy, evlx, evly, True)

eval(model, model_2, trainx, trainy, testx, testy, False)
