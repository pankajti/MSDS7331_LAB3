
from surprise import Dataset, SVD, KNNBaseline, KNNBasic, KNNWithMeans

# Use movielens-100K
data = Dataset.load_builtin('ml-100k')

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}

from surprise.model_selection import GridSearchCV

param_grid = {'k': [40, 100], 'min_k': [5, 10],}

gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

be = gs.best_estimator

print(be)