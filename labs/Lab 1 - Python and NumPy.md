---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
**Ben Cahill**


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


**YOUR ANSWER HERE**


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.

```python
import numpy as np
```

## Exercise 1


#### Make an array a of size 6 × 4 where every element is a 2

```python
a = [[2 for i in range(6)] for j in range(4)]
a = np.ones((6,4),dtype = int) * 2
a = np.full((6,4), 2)
a
```

## Exercise 2


#### Make an array b of size 6 × 4 that has 3 on the leading diagonal and 1 everywhere else. (You can do this without loops.)


```python
b = np.full((6,4), 1)
np.fill_diagonal(b,3)
b

# also could have done b[range(4),range(4)] = 3
```

## Exercise 3


#### Can you multiply these two matrices together? Why does a * b work, but not dot(a,b)?

```python
a * b
```

```python
# np.dot(a,b)
```

Using the dot product produces an error because the matrixes have to have the correct dimentions (the inner dimetions need to be the same) in order to calculate the dot product. a * b is element by element multiplaction.


## Exercise 4


#### Compute dot(a.transpose(),b) and dot(a,b.transpose()). Why are the results different shapes?

```python
np.dot(a.transpose(),b)
```

```python
np.dot(b.transpose(),a)
```

The shapes are different because the output of a dot product is always equal in dimention to the outer dimentions of the two matrixes.


## Exercise 5


#### Write a function that prints some output on the screen and make sure you can run it in the programming environment that you are using

```python
def print_stuff(stuff):
    print(stuff)
    
print_stuff("stuff")
```

## Exercise 6


#### Now write one that makes some random arrays and prints out their sums, the mean value, etc.

```python
def array_func():
    
    rand = np.random.randint(1,10,15)
    print("Array ",rand)
    
    sum = np.sum(rand)
    print("Sum ", sum)
    
    mean = np.mean(rand)
    print("Mean ",mean)
    
```

```python
array_func()
```

```python
array_func()
```

## Exercise 7


#### Write a function that consists of a set of loops that run through an array and count the number of ones in it. Do the same thing using the where() function (use info(where) to find out how to use it)

```python
def count_ones(arr):
    ones = 0
    
    for item in arr:
        if item == 1:
            ones += 1
            
    return ones

# for 2D, use arr.shape[0] to get items in rows, arr.shape[1] to items in get columns using numpy
```

```python
rand = np.random.randint(1,10,15)

print(rand)
count_ones(rand)
```

```python
print(rand)
len(np.where(rand == 1)[0])

# for 2D, returns a tuple (which cannot be modified)
# for 2D can use zip in python to unzip the arguments of the tuple to return the index (x,y) locations
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.


#### Make an array a of size 6 × 4 where every element is a 2

```python
import pandas as pd

pd_a = pd.DataFrame(np.full((6,4), 2))
pd_a
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.


#### Make an array b of size 6 × 4 that has 3 on the leading diagonal and 1 everywhere else. (You can do this without loops.)

```python
b = np.full((6,4), 1)
np.fill_diagonal(b,3)
pd_b = pd.DataFrame(b)
pd_b

# or b.iloc[range(4),range(4)] = 3
# b.values to convert to np array
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.


#### Can you multiply these two matrices together? Why does a * b work, but not dot(a,b)?

```python
mult = pd_a * pd_b
mult
```

```python
# np.dot(pd_a,pd_b)
```

Using the dot product produces an error because the matrixes have to have the correct dimentions (the inner dimetions need to be the same) in order to calculate the dot product. a * b is element by element multiplaction.


## Exercise 11
Repeat exercise A.7 using a dataframe.


#### Write a function that consists of a set of loops that run through an array and count the number of ones in it. Do the same thing using the where() function (use info(where) to find out how to use it)

```python
#currently only works for dataframes with a single column, but could be modified to work with other types of dfs
def count_ones(df):
    arr = np.array(df)
    ones = 0
    
    for item in arr:
        if item == 1:
            ones += 1
            
    return ones
```

```python
rand_df = pd.DataFrame(np.random.randint(1,10,15))
```

```python
print(rand_df)
count_ones(rand_df)
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
titanic_df["name"]
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
titanic_df.set_index('sex',inplace=True)
```

```python
len(titanic_df.loc["female"])
```

## Exercise 14
How do you reset the index?

```python
titanic_df.reset_index()
```
