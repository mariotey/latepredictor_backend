# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from .automodels import (
    AutoLinearRegression,
    AutoRidge,
    AutoRandomForest,
    AutoGBoost
)


LINEAR_MODELS = [
    # ("linear_regression", LinearRegression()),
    ("linear_regression", AutoLinearRegression()),
    # ("ridge", Ridge(alpha=1.0)),
    ("ridge", AutoRidge()),
]


TREE_MODELS = [
    # ("random_forest", RandomForestRegressor()),
    ("random_forest", AutoRandomForest()),
    # ("gboost", GradientBoostingRegressor()),
    ("gboost", AutoGBoost()),
]