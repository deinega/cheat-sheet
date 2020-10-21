
### MATPLOTLIB

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

plt.imshow(X_train[1, ;, :0], cmap="Greys")

plt.plot(df.x, df.y) # line
plt.scatter(df.x, df.y,  color='b', marker='o', s=1, alpha=.2, label='aapl')
plt.countour(x, y, z, lines) # plt.colorbar()
plt.hist(sample, bins=np.linspace(-3,3,20), normed=True)
plt.bar(x1, y1, alpha=.9, color='r')
plt.bar(x2, y2, alpha=.6, color='b')
plt.pie(x, labels=labels) # add plt.axis('equal') if subplots

plt.xlim(x_min, x_max)
plt.xticks(np.linspace(0,10,1))
plt.xlabel(name)
plt.title(name)
plt.legend()
plt.grid()
plt.axes().set_aspect(1)
plt.savefig(filename)

plt.show()

fig, axes = plt.subplots(2, 2, figsize=(7, 7))

plt.subplot(2, 1, 1)
plt.subplot(2, 1, 2)

plt.hist(x = x, bins = 15)

fig = plt.figure(figsize=(x,y))
ax = fig.add_subplot(111, projection='3d')
ax.set...
scatter = ax.scatter(X, Y, Z) # we are manipulating object instead of using plt

from matplotlib import style
style.use('ggplot')

from matplotlib import animation
anim = animation.FuncAnimcation(fig, anumate, init_func=init, frames=100, interval=20, blit=True)
anim.to_html5_video()

from mpl_toolkits.mplot3d import Axes3D

### SEABORN

import seaborn as sns

sns.set() # make matplotlib plots prettier

sns.countplot(df['target'])

sns.scatterplot(x=df['Height'], y=df['Wingspan'], hue=df['Gender'])

sns.distplot(tips_data["total_bill"], kde=False, fit=norm).set_title("Histogram of Total Bill")
plt.show()

sns.boxplot(tips_data["total_bill"])
sns.boxplot(tips_data["tip"]).set_title("Box plot of the Total Bill and Tips")
plt.show()

sns.boxplot(data=df.loc[:, ["Age", "Height", "Wingspan", "CWDistance", "Score"]])
plt.show()

sns.boxplot(x = tips_data["tip"], y = tips_data["smoker"])
plt.show()

g = sns.FacetGrid(tips_data, row = "day")
g = g.map(plt.hist, "tip")
plt.show()

g = sns.FacetGrid(df, col="Gender", hue="loan_status")
g.map(plt.hist, 'age')
plt.show()

