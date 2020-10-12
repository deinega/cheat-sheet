
### preprocessing

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

imputer = SimpleImputer(strategy='most_frequent')
imputer.statistics_ # call after fitting, return filled values (f.e., medians)

ordinal_encoder = OrdinalEncoder() # call for all 2D data
ordinal_encoder.categories_

laber_encoder = LabelEncoder() # call for each column, if number of categories is pre-defined
laber_encoder.fit(['F','M'])
label_encoder.classes_

encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) # requires input integers
encoder = LabelBinarizer()

X = StandardScaler().fit(X).transform(X.astype(float))
scaler = RobustScaler() # uses quantiles

poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)

CountVectorizer(ngram_range=1, analyzer='word', max_df) # bag of words

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, ...)
    def fit(self, X, y=None)
    def transform(self, X)

### pipelines

form sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer

categorical_transformer = pipeline.Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

pipeline = pipeline.Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

preprocessor = compose.ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

### train / validation

from sklearn.model_selection import ShuffleSplit, StratifiedShuffledSplit
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import learning_curve

split = StratifiedShuffledSplit(n_splits=1, test_size=.2, random_state=0)
train_index, test_index = split.split(df, df['income_cat'])
strat_train_set = housing.loc[train_index]
strat_test_set = housing.loc[test_index]

kf = KFold(shuffle=True, n_splits=5, random_state=0)
skf = StratifiedKFold(n_splits=3)

for train_ind, test_ind in skf.split(X_train):
    ...

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scores = -1 * cross_val_score(regr, x_train, y_train, cv=5, scoring='neg_mean_squared_error') # you can use fold object as cv
scores.mean()

scores = cross_validate(model, X, y, cv=5, scoring={'mean_squared_error': metrics.make_scorer(mean_squared_error)}) # multiple scorings, returns dictionary

# scoring='accuracy', 'precision', 'recall_macro', 'f1_macro' 
# for multilabel: average='macro'

y_hat = cross_val_predict(clf, X_train, y_train, cv=3, method='predict') # method can be predict_proba

sizes, train_scores, test_scores = learning_curve(regressor, X, y, cv=5, train_sizes=train_sizes, scoring='r2')

### non-linear fitting

from scipy.optimize import curve_fit

popt, pcov = curve_fit(sigmoid, x_train, y_train) # popt - optimal parameters

### model

from sklearn.base import clone

model.fit(x_train, y_train)
model.score(X_test, y_test)
y_test = model.predict(x_test)

model.feature_importances_

with open ('clf.pickle', 'wb') as f:
    pickle.dump(clf, f)

clf = pickle.load(open('clf.picle', 'rb'))



from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression, SGDRegressor, SGDClassifier
from sklearn.svm import LinearSVR, SVR, LinearSVC, SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

regr = Ridge(alpha=1) # square
regr = Lasso(alpha=1) # module
regr.intercept_, regr.coef_

sgd_reg = SGDRegressor(n_iter=50, eta0=.1, penalty='l2') 
# warm_start=True if you want to call it multiple times, maybe for early stopping
sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True, penalty=None, learning_rate='constant', eta0=0.0005) 
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3) # hinge loss is equivalent to SVM, log loss gives logistic regression

svm = SVR() # epsilon hyperparameter
svm_reg = LinearSVR() # the same as SVR just linear kernel
svm_clf = SVC(kernel='rbf',C=1)
# SVM does not require much tuning
# C, alpha, lambda
# C: start with small and increase (1e-6 - ... *10). the larger C the longer takes
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# linear: try C; rbf: try C and gamma

tree_reg = DecisionTreeRegressor()
tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
# play with max_leaf_nodes (100), min_samples_split, min_samples_leaf, max_deapth
# params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}

LR = LogisticRegression(C=1) # multi_class='multinomial'
y_hat_prob = LR.predict_proba(x_test)
y_scores = LR.decision_function(x_test) # returns confidence score (signed distance of that sample to the hyperplane) to be used under threshold

neigh = KNeighborsClassifier(n_neighbors = 5) 
# param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}

gnb = GaussianNB() # works for real input

from sklearn.base import BaseEstimator, ClassifierMixin

class Never5Classifier(BaseEstimator[, ClassifierMixin]):
    def __init__(self, ...)
    def fit(self, X, y=None)
    def predict(self, X)

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# one to many, one to one (better for SVM since you can use smaller data set even if there are more classifiers)
# sklearn does it under the hood, or you can redefine it using classes above

### ensembles

from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

clf = VotingClassifier(estimators=[...], voting='hard')

bag_clf = (DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True)

rnd_clf = RandomForestClassifier()
# play with n_estimators (100), criterion ('mae'), min_samples_split (20), max_depth (7)
# green:
#  n_estimators (always good, 10 - ..., saturation of accuracy after n_estimators increases)
#  max_depth (7 - 10 -20 ..., could be None)
#  max_features 
# red:
#  mean_samples_leaf
# others:
#  criterion. gini (usually better) / enthropy
#  n_jobs

ada_clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=200)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=.1)

### metrics

from sklearn.metrics import mean_squared_error, r2_score, log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score

mse = mean_squared_error(y_test, y_hat) # np.mean((y_hat - y_test) ** 2)
accuracy = accuracy_score(y_test, y_hat)
precision = precision_score(y_test, yhat, average='micro')
recall = recall_score(y_test, yhat, average='macro')
f1 = f1_score(y_test, yhat, average='weighted')

cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
print(classification_report(y_test, yhat))

y_scores = sgd_clf.decision_function([some_digit]) # returns confidence score, and later you can choose your own threshold
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
score = roc_auc_score(y_test, y_hat)

### dimensionality reduction

from sklearn.decomposition import PCA, KernelPCA

pca = PCA(n_components=20)
X_all = np.concantenate([X_train, X_test])
pca.fit(X_all)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

pca.explained_variance_ratio_, components_

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

### clustering

from sklearn.cluster import KMeans, MeanShift, DBSCAN

k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)
k_means.fit(X)
k_means.predict(X)

k_means.labels_, cluster_centers_

dbscan = DBSCAN(eps=0.05, min_samples=5)

### search for hyperparameters

from scipy.stats import randint, uniform, expon, reciprocal
from sklearn.model_selection import GridSearchCV

model = SVC()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# scoring can be make_scorer(mean_squared_error)
grid_search = GridSearchCV(model, param_grid=parameters, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=10) # cv is not neccessary

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
        'C': reciprocal(20, 200000), # you don't know the scale of hyperparameters
        'gamma': expon(scale=1.0), # you know the scale,
        'C': uniform(1, 10),
        'kernel': ['linear', 'rbf'],
    }
RandomizedSearchCV(model, param_distributions=param_distribs, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=0)

clf.fit(X, y)
clf.best_estimator_
clf.best_params_
clf.best_score_
clf.cv_results_

### feature selection

from sklearn.feature_selection import SelectKBest, chi2, f_classif, SelectFromModel, RFE, RFECV

selector = SelectKBest(f_classif, k=5) # keep 5 features
train_new = selector.fit_transform(train[feature_cols], train['outcome'])
select_feature = SelectKBest(chi2, k=5).fit(x_train, y_train).scores_

estimator = LogisticRegression(C=1, penalty="l1", random_state=7).fit(X, y)
selector = SelectFromModel(estimator, prefit=True)
selector.get_support()
X_new = selector.transform(X)

estimator = RandomForestClassifier()
selector = RFE(estimator=estimator, n_features_to_select=5, step=1)
#selector = RFECV(estimator, step=1, scoring='accuracy', cv=5)
selector = selector.fit(X, y)
selector.get_support() # or .support_

### gradient boosting

model = xgboost.XGBRegressor(n_estimators=500, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)], learning_rate=0.1)

import lightgbm as lgb

feature_cols = train.columns.drop('outcome')

dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])

param = {'num_leaves': 64, 'objective': 'binary'}
param['metric'] = 'auc'
num_round = 1000
bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)

model = lgb.LGBMRegressor(...)

# hyperparameters (XGBoost / LightGBM): 
# green:
#  max_depth (7-30). if you increase the depth and can't get model to overfit, extract interactions
#  num_leaves (for LightGBM)
#  subsample/bagging_fraction. fraction of objects to split
#  cosample_bytree, cosample_bylevel / feature fraction. fraction of features to split
#  eta, num_rounds / learning_rate, num_iterations. free eta to be small (0.01, 0.1) and find num_rounds to overfit. after that multilplu num_rounds on alpha and divide eta on alpha (alha = 2). 
# red:
#  min_child_weight (!, 0-5-15-300), lambda, alpha / min_data_in_leaf, lambda_l1, lambda_l2


from catboost import CatBoostRegressor, Pool

### Vowpal Wabbit

FTRL, linear model not loading everything to memory

### smart categorical encoding

import category_encoders as ce

cat_cols = ['bank_name_clients', 'bank_branch_clients']
encoder = ce.OrdinalEncoder(cols=cat_cols)
loan_demographics = encoder.fit_transform(loan_demographics)

one_hot_enc = ce.OneHotEncoder(cols=cats)
loan_demographics = one_hot_enc.fit_transform(loan_demographics)

hash_enc = ce.HashingEncoder(cols=cat_cols, n_components=10)
loan_demographics = hash_enc.fit_transform(loan_demographics)

count_enc = ce.CountEncoder(cols=cat_features) # count encoding (effectively group rare categories with similar counts)
target_enc = ce.TargetEncoder(cols=cat_features) # target encoding
cat_boost_enc = ce.CatBoostEncoder(cols=cat_features) # CatBoost encoding


