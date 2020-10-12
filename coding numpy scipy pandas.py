
### NUMPY

np.array(list)
np.array(DataFrame)
np.arange(0., 10., .1)
np.meshgrid(x_arr, y_arr)
np.linspace(start, end, num)
np.zeros((size1, size2))
np.ones((size1, size2),dtype=np.int64)
np.full((size1, size2),3)

np.random.random((size1,size2)) # similar to np.random.rand
np.random.rand(size1,size2), np.random.randn(size1,size2)
np.random.permutation(size)

dtype, ndim, shape # rows, columns if ndim = 1 
astype
reshape, flatten
transpose
insert, delete, append
copy

len(arr)

np.concatenate
np.r_, np.c_
np.vstack((n1,n2)), hstack

# produces view, not new array, so you can rewrite it
[:] # row
[:,:] # row, column
[::-1]
[[...],:]
a[[True,False]]
a[:,np.newaxis]
a.squeeze(axis=None)
arr[arr>30] # 1-dimensional array

iterating through row or by index in range(len(arr))

+ - * / **
np.dot
np.sum, min, max, mean, median, mode (returns Series), std, argmax, any, all # axis=None, 0 or 1 (contraction over columns or rows), (2,3)
np.power, exp

np.where
np.argsort

y = np.linalg.solve(A, z)

### SCIPY

from scipi import optimize
optimize.minimize(f, x0=0)

from scipi import integrate
res, err = integrate.quad(f, 0, np.inf)

from scipi import interpolate
interpolate.interp1d(x, y, kind='quadratic', fill_value='extrapolate')

from scipy import stats

stats.norm.rvs(size=10) # normally distributed sample
stats.t.rvs(10, size=100)
stats.norm.pdf(x), cdf(x) # probability distribution, cumulative functions

stats.describe(x) # calculate statistics for sample

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(), scale=stats.sem(squared_errors))) # calculate confidence interval for the test RMSE
# this is equivalent to:
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt) # helps to check if distribution is normal
plt.show()

### STATSMODEL

import statsmodels.api as sm

sm.tsa.seasonal_decompose(wine.sales).plot()
sm.tsa.stattools.adfuller(wine.sales)[1]

wine['sales_box'], lmbda = stats.boxcox(wine.sales)

sm.graphics.tsa.plot_acf(otg1diff.values.squeeze(), lags=25, ax=ax1)
sm.graphics.tsa.plot_pacf(otg1diff, lags=25, ax=ax2)

sm.graphics.tsa.plot_acf(wine.sales_box_diff2[13:].values.squeeze(), lags=48, ax=ax) # or pacf

from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA, ARMA, ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX

model = SARIMAX(wine.sales_box, order=(p, d, q), seasonal_order=(P, D, Q, 12), enforce_stationarity=False, enforce_invertibility=False))
model = SARIMAX(data1, exog=data2, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model = sm.tsa.ARIMA(src_data_model, order=(1,1,1), freq='W').fit(full_output=False, disp=0)
mode.fit(disp=-1)
model.aic
model.resid
model.summary()
pred = model.predict(start=176, end=211)
pred = model.predict('2013-05-26','2014-12-31', typ='levels')

### PANDAS

pd.set_option('display.max_columns', 100)

df = pd.read_csv("file.csv", delimiter=",",  nrows=1000000, index_col=0, parse_dates=['month'], dayfirst=True, skiprows=1, dtype={'a': np.int16, 'b': np.float64})
df.to_csv(index=False)

pd.Series(np_array/list, index=['1st', '2nd']) # works similar to numpy array (operations, indexation)
sort_index, sort_values(by=0, ascending=True)

pd.DataFrame(np_array/list, columns=prev_df.columns)
pd.DataFrame({'a': pd.Series, 'b': pd.Series})

# same functions could be applied to Series or DataFrame (f.e., converting them toSeries)

df.shape
df.index
df.columns
df.dtypes
df.values

df.head(10)
df.tail()
df.sample(weight=sample_weights)

len(df)

df.info() # indices, columns, types, etc
df.describe(include=['object', 'bool'])
df.count(), sum, min, max, ... # can be applied to columns as well: df['smth'].mean()
df.memory_usage()

df.corr()
pd.crosstab(df.column2, df.column2)

df['width'].value_counts(normalize=False)
df['a'].unique()
df['a'].nunique(dropna=False) # number of unique

numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
df.select_dtypes(exclude=['objects'])

df.loc[0:5, 'State':'Area code']
df.iloc[[True, False, True], 0:3]

s[3:1:-1]
df['width']
df['a':'b']
df[['width','length']]

df['Churn'] = df['Churn'].astype('int')
df['due_date'] = pd.to_datetime(df['due_date'], format='%d.%m.%Y') (or df.index)
df_new["Year"] = df_new["Date"].dt.year
pd.to_numeric(df['a'])

df['Total'] = df['Day'] + 2 * df['Night']
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
# map - Series, apply - Series or DataFrame, applymap - DataFrame

df[(df['Churn'] == 1) & (df.price > 3)] # filter rows, works like df[[True, True, False]]
sales[sales['shop_id'].isin([26, 27, 28])]

df.isnull()
df.isnull().any().describe()
df.isna().mean()
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
incomplete_rows = housing[housing.isnull().any(axis=1)]

df.drop([1, 2]) # drop rows
df.drop(['bedroom'], axis=1) # drop columns

df.dropna(axis=0, subset=['bedrooms']) # drop rows
df.dropna(axis=1) # drop columns

df['bedrooms].fillna(df["bedrooms"].median())

df.replace({'NAN': 0})
X['Title'].replace(['Don', 'Dona', 'Rev'], 'Special')

df['forecast'].shift(-10)

df.assign(hour=df.launched.dt.hour, day=df.launched.dt.day)

X['item_cnt_month'] = X['item_cnt_month'].clip(0,20)
X['AgeBand'] = pd.cut(X['Age'], 5) # after that replace it with ordinals
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5]) # do for StratifiedShuffleSplit
X['CatAge'] = pd.qcut(X.Age, q=4, labels=False)

pd.get_dummies(X, columns=cat_cols, drop_first=True) # one-hot encoding
pd.concat([df,pd.get_dummies(df['education'])], axis=1)
codes, uniques = pd.factorize(['b', 'b', 'a', 'c', 'b']) # label encoding, order of appearance

all_email = emails.set_index(['week', 'member']).reindex(complete_idx, fill_value = 0).reset_index()
agg_don = donations.groupby('member').apply(lambda df: df.amount.resample("W-MON").sum().dropna())

# https://towardsdatascience.com/pandas-groupby-aggregate-transform-filter-c95ba3444bbb

df.groupby('region').mean() # produces DataFrameGroupBy object
df.groupby('region').beer_servings.agg(['count', 'mean'])
df.groupby(df['Sales Rep'].str.split(' ').str[0]).size()
df_train.groupby('Survived').Fare.describe()
gk.get_group('Boston Celtics') # you also iterate by groups
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False) # as_index = MultiIndex or flattened out RangeIndex

sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})
df.groupby('Sales Rep').agg({'Order Id':'size', 'Val':['sum','mean']})

df['b'] = df.a.map(df.groupby('a').size()/len(df)) # frequency encoding
df[col+'_mean_target'] = df[col].map(df.groupby(col).target.mean())
all_data['item_target_enc'] = all_data.groupby('item_id')['target'].transform('mean') # tranform = grouby + merge

df.sort_values('click_time', ascending=True) # to create train set in the past and validation set in the future
df.sort_values(['Churn', 'Total day charge'], ascending=[True, False])

df.query('a<b')


pd.melt(df, id_vars=["ID"], value_vars=["Name", "Role"])
df.join(other, lsuffix='_caller', rsuffix='_other') # by indices
df.join(other.set_index('key'), on='key')
pd.merge(df, shops, on=['shop_id'], how='left') # by default, merges on index
pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)

pd.groupby('VisitNumber')['Department'].value_counts().unstack() # after that, you can use tfidf

df = df.resample('W', how='mean')
rolling_google = google.High.rolling('90D').mean()
microsoft_std = microsoft.High.expanding().std()

# PLOTTING

# https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html

import matplotlib.pyplot as plt

# missing values
sns.heatmap(titanic.isnull(), cbar=True)

# see distribution of values

df.hist(bins=50, figsize=(14,10))
df.hist(column='income')
df['Close'].plot() # see dependence on index
sns.distplot(df_train.Fare, kde=False, fit=norm)
sns.boxplot(x=train.item_cnt_day)
sns.countplot(x='Survived', data=df_train)
plt.xticks(rotation=45);

# pivot table

oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")

# see repeated values, index versus value

plt.plot(x, '.')
plt.scatter(range(len(x)), x, c=y)

# see correlations
# we can also check if train and test data have the same distribution
# it helps to generate new features as a relationship between x1 and x2

plt.scatter(x1, x2)

sns.barplot(x="Assortment", y="Sales", data=df)
sns.boxplot(x='Parch',y='Age', data=df, palette='hls')
sns.catplot(x='SibSp', y='Survived', data = titanic, kind = 'point', aspect=2)
sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train)

df_train.groupby('Survived').Fare.hist(alpha=0.6)
gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})
df.plot(x='longitude', y='latitude', kind='scatter', alpha=.1, s=housing["population"]/100, label="population", c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
sns.lmplot(x='Age', y='Fare', hue='Survived', data=df_train, fit_reg=False, scatter_kws={'alpha':0.5});
sns.pairplot(df_train, hue='Survived', height=2.5)

df.corr().style.background_gradient(cmap='coolwarm').set_precision(2)
df.corr['target'].sort_values(ascending=False)
sns.heatmap(df.corr())

for i, col in enumerate(cols):
	sns.distplot(df[col], color=random.choice(colors))
    sns.jointplot(x=col, y="PRICE", data=df)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)

from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(15, 15), diagonal='kde')

plt.matshow(...) # confusion matrix. statistics for pairs of features (how mamy times one feature > another feature). can be used to find out clusters

# time history

sns.pointplot(x='date_block_num', y='item_cnt_day', hue='item_category_id', 
                      data=train[np.logical_and(count*id_per_graph <= train['item_category_id'], train['item_category_id'] < (count+1)*id_per_graph)]


