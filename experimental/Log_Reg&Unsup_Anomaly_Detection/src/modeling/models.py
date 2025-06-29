from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score

def train_classification_pipeline(
    train_x, train_y, val_x, val_y,
    score_func=f_classif,
    C=1.0,
    solver='liblinear',
    penalty='l1',
    k_features=50,
    threshold=0.0
):
    pipeline = Pipeline([
        ('remove_constant', VarianceThreshold(threshold=threshold)),
        ('feature_selection', SelectKBest(score_func=score_func, k=k_features)),
        ('model', LogisticRegression(C=C, solver=solver, class_weight="balanced", penalty=penalty))
    ])

    pipeline.fit(train_x, train_y)

    y_pred = pipeline.predict(val_x)
    y_prob = pipeline.predict_proba(val_x)[:, 1]

    roc_auc = roc_auc_score(val_y, y_prob)
    precision = precision_score(val_y, y_pred)
    recall = recall_score(val_y, y_pred)

    print("Logistic Regression Classification Report:")
    print(classification_report(val_y, y_pred))
    print(f"ROC AUC: {roc_auc:.4f}")

    return pipeline, y_pred, roc_auc, precision, recall

def run_isolation_forest(
    train_x, train_y, val_x, val_y,
    contamination=0.5,
    n_estimators=100,
    max_samples='auto',
    max_features=1.0,
    bootstrap=False,
    n_jobs=None,
    random_state=42,
    verbose=0
):
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)

    x_train_normal = train_x[train_y == 0]

    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        bootstrap=bootstrap,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=verbose
    )

    model.fit(x_train_normal)

    y_pred = model.predict(val_x)
    y_pred = [0 if y == 1 else 1 for y in y_pred]

    scores = model.decision_function(val_x)
    roc_auc = roc_auc_score(val_y, -scores)
    precision = precision_score(val_y, y_pred)
    recall = recall_score(val_y, y_pred)

    print("Isolation Forest Classification Report:")
    print(classification_report(val_y, y_pred))
    print(f"ROC AUC: {roc_auc:.4f}")

    return model, y_pred, roc_auc, precision, recall

def run_one_class_svm(
    train_x, train_y, val_x, val_y,
    kernel='rbf',
    gamma='auto',
    nu=0.1
):
    train_x = train_x.reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)

    x_train_normal = train_x[train_y == 0]

    model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
    model.fit(x_train_normal)

    y_pred = model.predict(val_x)
    y_pred = [0 if y == 1 else 1 for y in y_pred]

    scores = model.decision_function(val_x)
    roc_auc = roc_auc_score(val_y, -scores)
    precision = precision_score(val_y, y_pred)
    recall = recall_score(val_y, y_pred)

    print("One-Class SVM Classification Report:")
    print(classification_report(val_y, y_pred))
    print(f"ROC AUC: {roc_auc:.4f}")

    return model, y_pred, roc_auc, precision, recall

