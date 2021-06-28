from collections import defaultdict

import numpy as np
import tensorflow as tf
import xgboost
from sklearn import metrics
from sklearn import svm, linear_model, ensemble, naive_bayes, tree, \
    discriminant_analysis, neural_network
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Flatten, Dense


def multi_layer_perceptron():
    "from https://www.tensorflow.org/tutorials/keras/classification"
    model = tf.keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


def prep_data(samples):
    """
    Transforms input samples of shape (n, 28, 28) and values between 0-1 or 0-255 to
    shape: (n, 784) and values between 0-1
    """
    assert np.min(samples) >= 0
    if np.max(samples) > 1:
        print("dividing test_images by 255 since max value is bigger than 1")
        samples = samples / 255

    samples = samples.reshape(-1, 28 * 28)
    return samples


def get_accuracies(gen_samples, gen_labels, real_samples, real_labels):
    print(f"\t\t Calculating downstream classifier accuracies, "
          f"{len(gen_samples)} training samples, {len(real_samples)} test samples")

    gen_samples = prep_data(gen_samples)
    real_samples = prep_data(real_samples)

    models, model_specs = get_classifiers()
    accuracies = {}

    for i, model_key in enumerate(models.keys()):
        print(f"\t\t Training and evaluating classifier {i + 1}/{len(models.keys())}: {model_key}", end="")
        model = models[model_key](**model_specs[model_key])

        model.fit(gen_samples, gen_labels)
        y_pred = model.predict(real_samples)
        test_acc = accuracy_score(real_labels, y_pred)
        print(f". accuracy: {test_acc}")

        accuracies[model_key] = test_acc

    return accuracies


def get_accuracies_ecg(train_samples, train_labels, test_samples, test_labels):
    print(f"\t\t Calculating downstream classifier accuracies, "
          f"{len(train_samples)} training samples, {len(test_samples)} test samples")

    # From https://github.com/astorfi/differentially-private-cgan
    sc = MinMaxScaler()

    X_train = sc.fit_transform(train_samples)
    X_test = sc.transform(test_samples)

    n_estimator = 100
    # cls = RandomForestClassifier(max_depth=5, n_estimators=n_estimator)
    cls = GradientBoostingClassifier(n_estimators=n_estimator)
    cls.fit(X_train, train_labels)
    y_pred = cls.predict(X_test)
    score = metrics.accuracy_score(test_labels, y_pred)

    y_pred = cls.predict_proba(X_test)[:, 1]
    fpr_rf_lm, tpr_rf_lm, _ = metrics.roc_curve(test_labels, y_pred)
    AUROC = metrics.auc(fpr_rf_lm, tpr_rf_lm)

    precision, recall, thresholds = metrics.precision_recall_curve(test_labels, y_pred)
    AUPRC = metrics.auc(recall, precision)

    f1 = metrics.f1_score(test_labels, cls.predict(X_test))

    return {"accuracy": score, "AUROC": AUROC, "AUPRC": AUPRC, "F1": f1}


def get_classifiers():
    """Inspired by https://github.com/frhrdr/dp-merf/blob/main/code_balanced/synth_data_benchmark.py"""

    models = {'logistic_reg': linear_model.LogisticRegression,
              'random_forest': ensemble.RandomForestClassifier,
              'gaussian_nb': naive_bayes.GaussianNB,
              'bernoulli_nb': naive_bayes.BernoulliNB,
              'linear_svc': svm.LinearSVC,
              'decision_tree': tree.DecisionTreeClassifier,
              'lda': discriminant_analysis.LinearDiscriminantAnalysis,
              'adaboost': ensemble.AdaBoostClassifier,
              'mlp': neural_network.MLPClassifier,
              'bagging': ensemble.BaggingClassifier,
              'gbm': ensemble.GradientBoostingClassifier,
              'xgboost': xgboost.XGBClassifier}

    slow_models = {'bagging', 'gbm', 'xgboost'}

    model_specs = defaultdict(dict)
    model_specs['logistic_reg'] = {'solver': 'lbfgs', 'max_iter': 5000, 'multi_class': 'auto'}
    model_specs['random_forest'] = {'n_estimators': 100, 'class_weight': 'balanced'}
    model_specs['linear_svc'] = {'max_iter': 10000, 'tol': 1e-8, 'loss': 'hinge'}
    model_specs['bernoulli_nb'] = {'binarize': 0.5}
    model_specs['lda'] = {'solver': 'eigen', 'n_components': 9, 'tol': 1e-8, 'shrinkage': 0.5}
    model_specs['decision_tree'] = {'class_weight': 'balanced', 'criterion': 'gini', 'splitter': 'best',
                                    'min_samples_split': 2, 'min_samples_leaf': 1, 'min_weight_fraction_leaf': 0.0,
                                    'min_impurity_decrease': 0.0}
    model_specs['adaboost'] = {'n_estimators': 100, 'algorithm': 'SAMME.R'}  # setting used in neurips2020 submission
    # model_specs['adaboost'] = {'n_estimators': 100, 'learning_rate': 0.1, 'algorithm': 'SAMME.R'}  best so far
    model_specs['bagging'] = {'max_samples': 0.1, 'n_estimators': 20}
    model_specs['gbm'] = {'subsample': 0.1, 'n_estimators': 50}
    model_specs['xgboost'] = {'colsample_bytree': 0.1, 'objective': 'multi:softprob', 'n_estimators': 50}

    return models, model_specs


def model_from_checkpoint(path, model):
    """
    Loads tensorflow model from checkpoint.
    @param path to directory containing checkpoint files
    @param model untrained model to add weights to
    """
    if not path.endswith("cp.ckpt"):
        path += "/cp.ckpt"
    model.load_weights(path)
    return model
