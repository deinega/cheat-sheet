
# general

getwd()
setwd('/home/alex/code')

opions(...) # digits, error, width
options(stringsAsFactors = FALSE)

source('file.R')

rm(list=ls())

.libPaths()
installed.packages()
library(package)
require(package)
install.packages('package', dependencies = TRUE)
update.packages()

library('devtools')
devtools::install_github(package)

devtools # installing packages from github
vroom # reading csv, can read from internet, can read number of files at once
stringr # strings
dplyr # manipulation with dataframes
tidyr # 
ggplot2 # visualization
forecast

sessionInfo()

# types

typeof(x1)
is.double(x)
is.null(l$string)
as.double(x)
as.Date(date, '%d %m %Y')

# conditions, loops

if {} else {}
if_else(x>1, 'high', 'low') # vector version of if. can be embedded

repeat {}
while {}
for (i in 1:8) {} # next, break

# functions

f <- function(x){
	y <<- x+1 # to global environment
	return x+1 # return ca be omitted
}

%>% # pipeline to avoid calling function inside the function. can use . in the next function to specify where to put result

replicate(5, fun(100))

outer(letters, LETTERS, paste0)
Vectorize(f, 'arg')

do.call(rbind, list(df1, df2, df3))

# S3 (method dispatch, generic functions)

z <- list(...)
class(z) = 'Z'

methods(print)
print.Z <- function(z) {...}

grade <- function(obj) { # generic function, such as print
	UseMethod("grade")
}

# S4

setClass('Z', representation(name='character', salary='numeric'))
z = new('Z', name='Joe', salary=100)
z = Z(name='Joe', salary=100)
z@name
slot(z, name)

showMethods(show)
setMethod('show', 'Z', function(object){...}) # generic S4 function, such as show

# Reference classes

student <- setRefClass("student", fields = list(name = "character", age = "numeric", GPA = "numeric"
s <- student(name = "John", age = 21, GPA = 3.5, methods = list(inc_age = function(x) {age <<- age + x}))
square <- Polygon$new(sides = 4)
s$name

Person$methods(say_hello = function() message("Hi!"))

s$copy()

# R6

factory <- R6Class('Thing', inherit = parent_factory, private = list(...), public = list(fun = function(){})) # use shared for static
factory$new()

# useful functions

all.equal(.1 + .05 == .15)

sort(x)
unique(x)
all, any
sum, max
which(1:10==5) # position of elements which is true
which.max # only one element

1:5 %in% c(1, 2, 5)

# data structures

vector(mode='character', length=2)
matrix(1:6, nrow=2, ncol=3, byrow=FALSE)
list(name = 'john', 1, vec, c(l1, l2))
new.env()
data.frame(x = 1:4, b = c(T,F), row.names = c(1, 2, 3, 4))

## vector creating

5:8
c(5, 8)
c(uno = 1, dos = 2, 3)

seq(1, 2, by=.25)
rep(x, times=3)
rep(x, each=4)
set.seed(42)
sample(1:100, 50, replace = TRUE)

vec = unlist(l)

## indexing

x[-c(1,3)]
x(c(TRUE, FALSE))
x(x > 1 & x < 3)
x(c('uno', 'dos'))

m[2,]
m[, c(1,3,5)]

l[1]
l[[1]] # returns one element
l$node
l['node']

df$column # returns vector
df['column'] # returns subtable
df[3:4, -1]
df[c(F, T), c('x', 'z')]
df[, 1, drop = FALSE]
df[df$x > 2, ]
subset(df, x > 2, select = c(x, z))

## functions for structures

length(x) # can be changed
names(x) # can be changed
dim(x) # nrow, ncol
rowSums, colMeans
rownames, colnames # for dataframes
rbind, cbind

%*%

attr(x, 'author') <- 'Caesar' # for lists
attr(x, 'names')
attributes(x)

# dataframes

read.table(file='name', header=TRUE, sep=',') # na.strings, colClasses, skip
read.csv # different defailt values
write.table
write.csv

str, summary, head, tail

df[complete.cases(df)]
any(!complete.cases(df))
na.omit(df)

df$var <- NULL

within(df, a, b)

merge(df, df_salary, by = 'x')

# dplyr

filter(df, column == 'column_value' & column2 > 10 & column3 %in% c(1, 2, 3))
select(df, column1, column2)
select(df, column1:column5)
select_at(df, vars(matches('^s')))
select_if(df, is.numeric)
arrange(df, month)
rename(df, column_new_name=column_old_name) # there are rename_if, rename_at, rename_all
mutate(df, c = a/b) # mutate_if, mutate_at. transmute remove all columns besides specified
mutate(group_by(department), total_dep = sum(total)) # does not reduce number of rows
summarize(group_by(df, date), sessions = sum(sessions)) # sumamrize_if, sumamrize_at
summarize_at(group_by(df, date, medium), c('sessions', 'bounces'), min)
summarize_if(group_by(df, date, medium), is.numeric, list(avg=mean, count=length), na.rm=TRUE) # avg, cout are suffces for resulted colunns

bind_rows(df1, df2)
left_join(df1, df2, by=c('id'='employe_id', 'month')) # semi_join, anti_join
distinct(df)

# tidyr

fill(df, region, .direction='down')
pivot_longer(df, cols='jan:dec', names_to='month', values_to='sales')
separate(df, col='month', into=c('month', 'year'), remove=TRUE, sep=' ')
pivot_wider(df, names_from=year, values_from=sales)

build_wider_spec # store transformation
pivot_wider_spec # apply stored transformation

# factors

f = factor(c('a', 'b', 'a', 'c')) # numeric and character at the same time
f = ordered(c('a', 'b', 'a', 'c'), c('a', 'b', 'c'))
df$a <- as.factor(df$a)

levels(f)
levels[f][1] <- = 'bbb'
nlevels(f)
droplevels(f) # remove empty levels

table(cut(norm(10, -5:5)))

# apply

apply(xm, 1, f)
mapply(seq, from=1:4, to=2:5, by=.8)
lapply(l, length)  # sapply simplifies it to vector
lapply(l, paste, collapse='|')
tapply(df$factor, df$number, max)

# strings

paste(c('a', 'b'), 'x', sep='_')
paste(c('a', 'b'), 'x', colapse=',')
strsplit(s, ' ', fixed = TRUE) # if fixed is FALSE then string is considered as regular expression
grep('work', s) # to find, grepl returns vector of booleans
gsub('work', '###', s) # to change
tolower, toupper

library(stringr)
str_extract # _all can be added to the function name
str_replace

format(date, '%d %B %Y')
formatC(pi, digits=3, format="e")

# files

dir(pattern='.*\\.csv$')
list.files(pattern='.*\\.csv$')
list.dirs('..', recursive = FALSE)

readLines('file.csv', 5)

# ggplot2

qplot(x=month, y=n, data=df, fill='darkcyan', geom='col', main='title', xlab='month', ylab='sales') # fill=shop. group='shop'
# geom=c('line', 'point'), c('boxplot'), c('histogram')


####

y <- ts(c(123,39,78,52,110), start=2012)
y <- ts(z, start=2003, frequency=12)

# plot
autoplot(melsyd[,"Economy.Class"]) +
  ggtitle("Economy class passengers: Melbourne-Sydney") +
  xlab("Year") +
  ylab("Thousands")


# histogram
gghistogram(res) + ggtitle("Histogram of residuals")

# seasonal plot
ggseasonplot(a10, year.labels=TRUE, year.labels.left=TRUE) +
  ylab("$ million") +
  ggtitle("Seasonal plot: antidiabetic drug sales")

# scatterplot
qplot(Temperature, Demand, data=as.data.frame(elecdemand)) +
  ylab("Demand (GW)") + xlab("Temperature (Celsius)")

# multiple plots
autoplot(visnights[,1:5], facets=TRUE) +
  ylab("Number of visitor nights each quarter (millions)")

# scatterplot matrix
GGally::ggpairs(as.data.frame(visnights[,1:5]))

# lag plot
gglagplot(beer2)

# auto-correlation
ggAcf(beer2)

# naive methods
meanf(y, h)
naive(y, h)
snaive(y, h)
rwf(y, h, drift=TRUE)
rwf(eggs, drift=TRUE, lambda=0, h=50, level=80, biasadj=TRUE)
forecast(ausbeer, h=4)

residuals(naive(goog200))

Box.test(res, lag=10, fitdf=0)
Box.test(res,lag=10, fitdf=0, type="Lj")
checkresiduals(naive(goog200))

accuracy(beerfit1, beer3)

e <- tsCV(goog200, rwf, drift=TRUE, h=1)
goog200 %>% tsCV(forecastfunction=rwf, drift=TRUE, h=1) -> e

window(ausbeer, start=1995)
subset(ausbeer, quarter = 1)
head(ausbeer, 4*5)
tail(ausbeer, 4*5)

dframe <- cbind(Monthly = milk, DailyAverage = milk/monthdays(milk))

fit.consMR <- tslm(
  Consumption ~ Income + Production + Unemployment + Savings,
  data=uschange)
summary(fit.consMR)

fit.beer <- tslm(beer2 ~ trend + season)
fourier.beer <- tslm(beer2 ~ trend + fourier(beer2, K=2))

# measures of predictive accuracy
CV(fit.consMR)

