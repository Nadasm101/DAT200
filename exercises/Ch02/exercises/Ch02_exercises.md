# Ch02 - Perceptron and Adaline algorithms

<img src="../../assets/workout.png"
     alt="Hierarchical clustering"
     style="width: 100px; float: right; margin: 20px" />

__Welcome to the practice exercises for chapter 02!__

The exercises are divided into two parts: A theoretical (quiz) part, and a practical part where you get to know the code behind the Perceptron and Adaline classes and try them in practice comparing results.

It is highly recommended to complete the quiz and understand the underlying theory __first__, before completing the more practical exercises.

__Enjoy!__


___


<img src="../../assets/quiz.jpg"
     alt="quiz"
     style="width: 100px; float: right" />

## Part I: Quiz-time <a class="anchor" id="quiz"></a>

__Question 1:__

Considering the following formulas:
$$w_{j} := w_{j} + \Delta w_{j}$$
$$where$$
$$\Delta w_{j} = \eta (y^{i} - \hat{y}^{j}) x_{j}^{(i)}$$

Explain the meaning of different symbols.

**Answer:**

$w_{j}$ : 

$\Delta w_{j}$ :

$y^{i}$ :

$\hat{y}^{i}$ :

$\eta$ :

$x_{j}^{(i)}$ : 

__Question 2:__

Calculate the $\Delta w_{j}$ for the following examples:

|task|$y^{j}$|$\hat{y}^{j}$|learning rate|$x_{j}^{(i)}$|$\Delta w_{j}$|
|---|---|---|---|---|---|
|A|1|1|0.1|0.5|??|
|B|-1|1|1|0.9|??|
|C|1|-1|2|1.2|??|
|D|-1|-1|0.05|0.05|??|


**Answer:**

A:

B:

C:

D:

__Question 3:__

Consider the following image:

<img src="images/perceptronadaline.png"
     alt="Perceptron and Adaline"
     style="width: 500px" />

Explain with your own words the key difference between the Perceptron and Adaline classifiers.

**Answer:**




___

<img src="../../assets/practice.png"
     alt="practice"
     style="width: 100px; float: right" />

## Part II:Complete the Percetron classifier

**Try to implement as much as possible without consulting the solution.**

*Purpose: Acquire a lasting understanding of basic and complex ML-algorithms by implementing weights, gradient descent and more by hand.*

**Resources:**
* Solution: Ch02 - page 27, 3rd Edition or in lecture notes


```python
class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        # Insert your code
        return

    def net_input(self, X):
        """Calculate net input"""
        # Insert your code
        return

    def predict(self, X):
        """Return class label after unit step"""
        # Insert your code
        return

        
```

---
<img src="../../assets/practice.png"
     alt="practice"
     style="width: 100px; float: right" />

## Part III: Complete the Adaline classifier

**Try to implement as much as possible without consulting the solution.**

**Resources:**
* Solution: Ch02 - page 27, 3rd Edition


```python
class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        # Insert your code
        return

    def net_input(self, X):
        """Calculate net input"""
        # Insert your code
        return

    def activation(self, X):
        """Compute linear activation"""
        # Insert your code
        return

    def predict(self, X):
        """Return class label after unit step"""
        # Insert your code
        return

```

---

<img src="../../assets/practice.png"
     alt="practice"
     style="width: 100px; float: right" />

## Part IV: Compare results on the Breast Cancer dataset

**Try to implement as much as possible without consulting the solution.**

**Resources:**
* Solution: Ch02 - page 27, 3rd Edition

Run the following code to load the Breast Cancer dataset:


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

# Import data
cancer_data = load_breast_cancer()

# Extract feature and target names
feature_names = cancer_data.feature_names
target_names = cancer_data.target_names

# Define feature data
X = cancer_data.data
X_std = StandardScaler().fit_transform(X)
print(f'Shape of X: {X.shape}')

# Define target data
y = cancer_data.target
# Convert y-valus to [-1 1]
y = np.where(y==1, 1, -1)
print(f'Shape of y: {y.shape}')

# Print distribution of targets
targets, counts = np.unique(y, return_counts=True)
target_count = {target_names[i]: counts[i] for i in range(len(targets))}
print(f'Distribution of targets: {target_count}')

# Divide training data into train and test splits
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.25, stratify=y, random_state=123)

print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of y_test: {y_test.shape}')
```

    Shape of X: (569, 30)
    Shape of y: (569,)
    Distribution of targets: {'malignant': 212, 'benign': 357}
    Shape of X_train: (426, 30)
    Shape of X_test: (143, 30)
    Shape of y_train: (426,)
    Shape of y_test: (143,)


==========================================================================

### A) Perceptron prediction

Train a perceptron using the Perceptron class above and find the F1-accuracy for the test set.


```python
# Create perceptron instance

pass

# Fit perceptron to training data

pass

# Make prediciton

pass

# Print f1-score for prediction

pass
```

==========================================================================

### B) Adaline prediction

**Train an adaline instance using the Adaline class above and find the F1-accuracy for the test set.**


```python
# Create perceptron instance

pass

# Fit perceptron to training data

pass

# Make prediciton

pass

# Print f1-score for prediction

pass
```

==========================================================================

### C) Tweak models and compare results

Try changing learning rates and iterations for both models. Which model is more precise? Which is more sensitive to input parameters?

**Answer:**

==========================================================================

### Bonus question:

Try running the Adaline model on the training data with the following parameters:
* eta: 0.1
* n_iter: 50
* random_state: 123

Comment on the result of the prediction. What do you think is the reason for the performance?

**Answer:**

___
___
