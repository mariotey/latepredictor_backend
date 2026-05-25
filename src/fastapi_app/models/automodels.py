import optuna
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer

N_TRIALS = 10
RANDOM_STATE = 42
KFOLD_N_SPLITS = 2
KFOLD_SHUFFLE_BOOL = True

SCORING = make_scorer(mean_squared_error, greater_is_better=False)


class AutoModel:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.best_score = None

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def search_space(self, trial):
        raise NotImplementedError

    def build_model(self, params):
        raise NotImplementedError

    def fit(self, X, y):

        cv = KFold(
            n_splits=KFOLD_N_SPLITS,
            shuffle=KFOLD_SHUFFLE_BOOL,
            random_state=RANDOM_STATE
        )

        def objective(trial):
            params = self.search_space(trial)
            model = self.build_model(params)

            score = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring=SCORING
            ).mean()

            return score

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(objective, n_trials=N_TRIALS)

        self.best_params = study.best_params
        self.best_score = study.best_value

        self.model = self.build_model(self.best_params)

        if hasattr(self.model, "fit"):
            self.model.fit(X, y)

        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict(X)


class AutoLinearRegression(AutoModel):
    def search_space(self, trial):
        return {
            "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        }

    def build_model(self, params):
        return LinearRegression(**params)


class AutoRidge(AutoModel):
    def search_space(self, trial):
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True),
        }

    def build_model(self, params):
        return Ridge(**params)


class AutoRandomForest(AutoModel):
    def search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 200),
            "max_depth": trial.suggest_int("max_depth", 5, 12),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 8),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
        }

    def build_model(self, params):
        return RandomForestRegressor(
            **params,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )


class AutoGBoost(AutoModel):
    def search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2),
            "max_depth": trial.suggest_int("max_depth", 2, 4),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        }

    def build_model(self, params):
        return GradientBoostingRegressor(
            **params,
            random_state=RANDOM_STATE
        )