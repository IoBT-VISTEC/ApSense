SVC_GRID_PARAMS = [
    {
        'clf__kernel': ['rbf'], 
        'clf__gamma': [1e-3, 1e-4],
        'clf__C': [0.001, 0.01, 0.1, 10, 50, 100],
        'clf__random_state': [42]
    },
    {
        'clf__kernel': ['linear'], 
        'clf__C': [0.001, 0.01, 0.1, 10, 50, 100],
        'clf__random_state': [42]
    }
]

RF_GRID_PARAMS = [
    {
        'clf__max_depth': [3, 5, 10],
        'clf__min_samples_split': [2, 5, 10]
    }
]

XGB_GRID_PARAMS = [
    {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [3, 6],
        'clf__learning_rate': [0.05, 0.1, 0.20]
    }
]

def get_params(model_name):
    d = {
        'SVC': SVC_GRID_PARAMS,
        'RF': RF_GRID_PARAMS,
        'XGB': XGB_GRID_PARAMS
    }
    return d[model_name]
