import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name} Results → RMSE: {rmse:.2f}, R²: {r2:.4f}")
    return rmse, r2

def train_and_compare_models(X_train,y_train, X_test, y_test):

    results = {}
    models = {}

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results['LinearRegression'] = evaluate_model("Linear Regression", lr, X_test, y_test)
    models['LinearRegression'] = lr

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    results['RandomForest'] = evaluate_model("Random Forest", rf, X_test, y_test)
    models['RandomForest'] = rf

    # 3. XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    results['XGBoost'] = evaluate_model("XGBoost", xgb, X_test, y_test)
    models['XGBoost'] = xgb

    # Identify best model by lowest RMSE
    best_model_name = min(results, key=lambda k: results[k][0])
    best_model = models[best_model_name]


    return best_model, best_model_name, results


def build_severity_models(params: dict, seed: int) -> dict:
    """Unfitted severity regressors keyed by name, using the values in params.yaml."""
    n_jobs = params['n_jobs']
    return {
        'mean_baseline': DummyRegressor(strategy='mean'),
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **params['random_forest']),
        'xgboost_default': XGBRegressor(random_state=seed, n_jobs=n_jobs, **params['xgboost_default']),
    }


def make_folds(n_splits: int, seed: int, groups=None):
    """Folds that keep every group (policy) inside one fold when groups are given."""
    if groups is None:
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=n_splits)


def tune_xgboost(X_train, y_train, params: dict, seed: int, groups=None) -> RandomizedSearchCV:
    """Randomized search over the XGBoost space, scored by cross-validated RMSE."""
    search = RandomizedSearchCV(
        XGBRegressor(random_state=seed, n_jobs=params['n_jobs']),
        param_distributions=params['search']['space'],
        n_iter=params['search']['n_iter'],
        scoring='neg_root_mean_squared_error',
        cv=make_folds(params['cv_folds'], seed, groups),
        random_state=seed,
        refit=True,
    )
    return search.fit(X_train, y_train, groups=groups)


def fit_severity_models(X_train, y_train, params: dict, seed: int, groups=None):
    """
    Fit every severity candidate and the tuned XGBoost model.

    Returns the fitted models, each model's cross-validated RMSE on the training
    data, and the winning XGBoost hyperparameters.
    """
    folds = make_folds(params['cv_folds'], seed, groups)
    models, cv_rmse = {}, {}
    for name, model in build_severity_models(params, seed).items():
        scores = cross_val_score(
            model, X_train, y_train, cv=folds, groups=groups, scoring='neg_root_mean_squared_error'
        )
        cv_rmse[name] = float(-scores.mean())
        models[name] = model.fit(X_train, y_train)

    search = tune_xgboost(X_train, y_train, params, seed, groups)
    models['xgboost_tuned'] = search.best_estimator_
    cv_rmse['xgboost_tuned'] = float(-search.best_score_)
    return models, cv_rmse, dict(search.best_params_)


REFERENCE_MODELS = ('mean_baseline',)


def select_best_model(cv_rmse: dict) -> str:
    """
    Name of the real model with the lowest cross-validated RMSE.

    Reference models such as the mean baseline are scored for comparison but are
    never selected. Ties go to the first name in sorted order.
    """
    candidates = [name for name in sorted(cv_rmse) if name not in REFERENCE_MODELS]
    return min(candidates, key=lambda name: cv_rmse[name])


def fit_claim_probability_models(X_train, y_train, params: dict, seed: int) -> dict:
    """Fit the logistic regression baseline and the XGBoost classifier."""
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=seed, **params['logistic_regression']),
    )
    boosted = XGBClassifier(random_state=seed, n_jobs=params['n_jobs'], **params['xgboost'])
    return {
        'logistic_regression': logistic.fit(X_train, y_train),
        'xgboost': boosted.fit(X_train, y_train),
    }
