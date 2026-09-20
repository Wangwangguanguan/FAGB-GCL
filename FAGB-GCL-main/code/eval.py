import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, dtype=bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def label_classification(embeddings, data):
    X = embeddings.detach().cpu().numpy()
    y_raw = data.y.detach().cpu().numpy()
    Y = y_raw.reshape(-1, 1)

    onehot_encoder = OneHotEncoder(categories='auto')
    Y = onehot_encoder.fit_transform(Y).toarray().astype(bool)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1, random_state=42, stratify=y_raw
    )

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(
        estimator=OneVsRestClassifier(logreg),
        param_grid=dict(estimator__C=c),
        n_jobs=8,
        cv=5,
        verbose=0
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    acc = accuracy_score(y_test, y_pred)
    return acc


def label_classification_with_target_nodes(embeddings, data, target_nodes=None):
    X = embeddings.detach().cpu().numpy()
    y_raw = data.y.detach().cpu().numpy()
    Y = y_raw.reshape(-1, 1)

    onehot_encoder = OneHotEncoder(categories='auto')
    Y = onehot_encoder.fit_transform(Y).toarray().astype(bool)

    X = normalize(X, norm='l2')

    idx_all = np.arange(X.shape[0])

    idx_train, idx_test, y_train_raw, y_test_raw = train_test_split(
        idx_all, y_raw, test_size=0.1, random_state=42, stratify=y_raw
    )

    X_train = X[idx_train]
    y_train = Y[idx_train]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(
        estimator=OneVsRestClassifier(logreg),
        param_grid=dict(estimator__C=c),
        n_jobs=8,
        cv=5,
        verbose=0
    )
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict_proba(X[idx_test])
    y_pred_test = prob_to_one_hot(y_pred_test)
    test_acc = accuracy_score(Y[idx_test], y_pred_test)

    target_acc = None
    if target_nodes is not None:
        target_nodes = np.array(target_nodes, dtype=np.int64)
        target_nodes = target_nodes[target_nodes < X.shape[0]]

        if len(target_nodes) > 0:
            y_pred_target = clf.predict_proba(X[target_nodes])
            y_pred_target = prob_to_one_hot(y_pred_target)
            target_acc = accuracy_score(Y[target_nodes], y_pred_target)

    return test_acc, target_acc