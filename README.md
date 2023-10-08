# SVM-Iris-dataset-gridsearchCV
## Advantages of Support Vector Machine:
### 1).SVM works relatively well when there is a clear margin of separation between classes.
### 2).SVM is more effective in high dimensional spaces.
### 3).SVM is effective in cases where the number of dimensions is greater than the number of samples.
### 4).SVM is relatively memory efficient.
## Disadvantages of Support Vector Machine:
### 1).SVM algorithm is not suitable for large data sets.
### 2).SVM does not perform very well when the data set has more noise i.e. target classes are overlapping.
### 3).In cases where the number of features for each data point exceeds the number of training data samples, the SVM will underperform.
### 4).As the support vector classifier works by putting data points, above and below the classifying hyperplane there is no probabilistic explanation for the classification.

## Hyper parameter tuning in svm using gridsearchcv
!https://miro.medium.com/max/1588/1*MHYG4D5Qixwpapha2aq4Pg.png

!https://miro.medium.com/max/1613/1*vbDxftix3ikkEzMwYbSdDw.png

In order to improve the model accuracy, there are several [parameters](https://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.svm.SVC.html) need to be tuned. Three major parameters including:

1. **Kernels**: The main function of the kernel is to take low dimensional input space
and transform it into a higher-dimensional space. It is mostly useful in non-linear separation problem.

!https://miro.medium.com/max/1380/1*AFqQio7ZB91FZvFtuYAn-w.png

2. **C (Regularisation)**:
 C is the penalty parameter, which represents misclassification or error
 term. The misclassification or error term tells the SVM optimisation 
how much error is bearable. This is how you can control the trade-off 
between decision boundary and misclassification term.

https://miro.medium.com/max/1000/0*08KrYhXpVQdUXWrX

when **C** is high it will classify all the data points correctly, also there is a chance to overfit.

3. **Gamma**: It defines how far influences the calculation of plausible line of separation.

!https://miro.medium.com/max/1713/1*6HVomcqW7BWuZ2vvGOEptw.png

when
 gamma is higher, nearby points will have high influence; low gamma 
means far away points also be considered to get the decision boundary.

## **Tuning the hyper-parameters of an estimator**

Hyper-parameters are parameters that are not directly learnt within estimators. In [scikit-learn](https://scikit-learn.org/stable/modules/grid_search.html), they are passed as arguments to the constructor of the estimator classes. **Grid search** is
 commonly used as an approach to hyper-parameter tuning that will 
methodically build and evaluate a model for each combination of 
algorithm parameters specified in a grid.

> GridSearchCV helps us combine an estimator with a grid search preamble to tune hyper-parameters.
> 

**Import GridsearchCV from Scikit Learn**

```
from sklearn.model_selection import GridSearchCV
```

**Create a dictionary called param_grid and fill out some parameters for kernels, C and gamma**

```
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
```

**Create a GridSearchCV object and fit it to the training data**

```
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
```

!https://miro.medium.com/max/1130/1*ZOA-zNkGitNvS0jejIvARg.png

**Find the optimal parameters**

```
print(grid.best_estimator_)
```

!https://miro.medium.com/max/1155/1*wp5gocIHfwAk00436rK0mg.png

found the best estimator using grid search

Take this grid model to create some predictions using the test set and then create classification reports and confusion matrices

```
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))#Output
[[15  0  0]
 [ 0 13  1]
 [ 0  0 16]]
```

!https://miro.medium.com/max/965/1*Bu4wAmB8E3hMFL51M7aS0g.png

For the coding and dataset, please check out [here](https://github.com/clareyan/SVM-Hyper-parameter-Tuning-using-GridSearchCV).
