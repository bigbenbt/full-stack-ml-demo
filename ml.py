from data_importer import import_data
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import numpy as np

df = import_data()
size = df.size


def main(current_feature):
    random_forest(current_feature)


def random_forest(current_feature):

    feature_list = df.columns
    labels = df[current_feature]
    features = df.drop(current_feature, axis=1)
    train_features, test_features, train_labels, test_labels = train_test_split(features,labels, test_size=0.25, random_state=37)
    rf = RandomForestRegressor(n_estimators=500, random_state=37)
    rf.fit(train_features, train_labels)
    predictions = rf.predict(test_features)

    errors = abs(predictions - test_labels)
    print('Average model error:', round(np.mean(errors), 2))

    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

    importances = list(rf.feature_importances_)

    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    gbm = GradientBoostingRegressor(n_estimators=4000, max_depth=6, alpha=0.1)
    gbm = gbm.fit(train_features, train_labels)

    preds2 = gbm.predict(X=test_features)

    errors2 = abs(preds2 - test_labels)
    print("average gradient boost error:", round(np.mean(errors2), 2))
    mape = 100 * (errors2 / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Gradient Boost Accuracy:', round(accuracy, 2), '%.')

    print("Feature importance from random forest: ")
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    return feature_importances


if __name__ == '__main__':
    main("medv")