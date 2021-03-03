# AI Programming - Assignment 1

<!--Delete the instructions below-->

## Getting started

1. To start clone this repository:

```bash
git clone git@github.com:ai-programming-21/assignment1-<your username here>.git
```

1. Complete the assignment and your report.
1. Commit and push your solution. It is advised to make a commit for each part of the solution.

```bash
git add .
git commit -m"describe the commit here"
git push origin main

```

<!--Delete the instruction above -->

Name: <!--Insert your name here -->

---

## Introduction

**NumPy** is a Python package. It stands for 'Numerical Python'. It is a library consisting of multidimensional array objects and a collection of routines for processing of array.

**NumPy** is often used along with packages like **SciPy** (Scientific Python) and **Matplotlib** (plotting library). This combination is widely used as a replacement for MatLab, a popular platform for technical computing.

**NumPy** and **SciPy** are open-source add-on modules to Python that provide common mathematical and numerical routines in pre-compiled, fast functions. These are growing into highly mature packages that provide functionality that meets, or perhaps exceeds, that associated with common commercial software like MatLab.

The **NumPy** (Numeric Python) package provides basic routines for manipulating large arrays and matrices of numeric data.

The **SciPy** (Scientific Python) package extends the functionality of **NumPy** with a substantial collection of useful algorithms, like minimization, Fourier transformation, regression, and other applied mathematical techniques.

However, Python as an alternative to MatLab is now seen as a more modern and has the advantage of being a complete programming language. It is open source, which is an added advantage of **NumPy**.

## Vectors

### Operations on vectors

#### Representing vectors

The simplest way to represent vectors in Python is using a list structure. A vector is constructed by giving the list of elements surrounded by square brackets, with the elements separated by commas.

The assignment operator = is used to give a name to the list. The len() function returns the size (dimension).

```python
x = [-1.1, 0.0, 3.6, 7.2]
len(x)
```

| Assignment |
| :--------- |
| Execute the Python code and print the value len(x). |
| **Solution:** |
| <!--Insert here -->|

Another common way to represent vectors in Python is to use a **numpy** array. To do so, we must first import the **numpy** package.

```python
import numpy as np
x = np.array([-1.1, 0.0, 3.6, 7.2])
len(x)
```

| Assignment |
| :--------- |
| Execute the Python code and print the value len(x). |
| **Solution:** |
| <!--Insert here -->|

#### Vector addition and substraction

If **x** and **y** are **numpy** arrays of the same size, **x+y** and **x-y** give their sum and difference, respectively.

```python
import numpy as np
x = np.array([1,2,3])
y = np.array([100,200,300])
print('Sum of arrays:', x+y)
print('Difference of arrays:', x-y)
```

| Assignment |
| :--------- |
| Execute the Python code. |
| What are the results of the two print statements. |
| Explain the results. |
| **Solution:** |
| <!--Insert here -->|

#### Scalar-vector multiplication and division

If *a* is a number and **x** is a **numpy** array (vector), you can express the scalar-vector product either as **a*x** or **x*a**.

```python
import numpy as np
x = np.array([1,2,3])
print('Sum of arrays:', 2.2 * x)

```

You can carry out scalar-vector division as **x/a**.

```python
import numpy as np
x = np.array([1,2,3])
print('Sum of arrays:', x / 2.2)

```

| Assignment |
| :--------- |
| Execute the Python code of both examples. |
| What are the results? |
| Explain the results. |
| **Solution:** |
| <!--Insert here -->|

#### Linear combination of vectors

You can form a linear combination in Python using scalar-vector multiplication and addition.

```python
import numpy as np
a = np.array([1,2])
b = np.array([3,4])
alpha = -0.5
beta = 1.5
c = alpha * a + beta * b 
print(c)

```

| Assignment |
| :--------- |
| Execute the Python code of both examples. |
| What is the value of c? |
| Explain the results. |
| **Solution:** |
| <!--Insert here -->|

To illustrate some additional Python syntax, we create a function that takes a list of coefficients and a list of vectors as its argument (input), and returns the linear combination (output).

```python
def lincomb(coef, vectors):
    n = len(vectors[0])
    comb = np.zeros(n)
    for i in range(len(vectors)):
        comb = comb + coef[i] * vectors[i]
    return comb

lincomb([alpha, beta], [a,b])
```

| Assignment |
| :--------- |
| Assign values to variables a, b, alpha and beta.|
| Execute the Python code and print the value of variable lincomb. |
| What is the result? |
| Explain the result. |
| **Solution:** |
| <!--Insert here -->|

#### Inner product

The inner product of n-vector x and y is denoted as **x<sup>T</sup> y**. 
In Python the inner product of x and y can be found using **np.inner(x,y)**

```python
import numpy as np
a = np.array([-1,2,2])
b = np.array([1,0,-3])
print(np.inner(x,y))

```

| Assignment |
| :--------- |
| Execute the Python code and print the value of the inner product.|
| What is the result? |
| Explain the result. |
| **Solution:** |
| <!--Insert here -->|

#### Checking properties

Let's check the distributive property *β.(a+b)= β.a+ β.b* which holds for any two n-vector a and b, and any scalar β.

```python
import numpy as np
a = np.random.random(3)
b = np.random.random(3)
beta = np.random.random()
lhs = beta*(a+b)
rhs = beta*a + beta*b
print('a:',a)
print('b:',b)
print('beta:',beta)
print('LHS:',lhs)
print('RHS:',rhs)
```

| Assignment |
| :--------- |
| Execute the Python code.|
| What is the result? |
| Explain the result. |
| **Solution:** |
| <!--Insert here -->|

## Linear functions

### Functions in Python

Python provides several methods for defining functions. One simple way is to use lambda functions. A simple function given by an expression such as **f(x)=x<sub>1</sub>+x<sub>2</sub>-x<sup>2</sup><sub>4</sub>** can be defined in a single line.

```python
f = lambda x: x[0] + x[1] - x[3]**2
f([-1,0,1,2])
```

| Assignment |
| :--------- |
| Execute the Python code and print the value of variable f.|
| What is the result? |
| Explain the result. |
| **Solution:** |
| <!--Insert here -->|

### Average function

Let's define the average function in Python and check its value of a specific vector. (Numpy also contains an average function, which can be called with np.mean).

```python
avg = lambda x: sum(x)/len(x)
x = [1,-3,2,-1]
avg(x)
```

| Assignment |
| :--------- |
| Execute the Python code and print the value of variable f.|
| What is the result? |
| Explain the result. |
| **Solution:** |
| <!--Insert here -->|

## Matrices

### Matrix addition

In Python, addition and subtraction of matrices, and scalar-matrix multiplication, both follow standard and mathematical notation.

```python
U = np.array([[0,4], [7,0], [3,1]])
V = np.array([[1,2], [2,3], [0,4]])
U+V
```

| Assignment |
| :--------- |
| Execute the Python code.|
| What is the result? |
| Explain the result. |
| **Solution:** |
| <!--Insert here -->|

### Inverse

If A is invertible, its inverse is given by np.linalg.inv(A). 
You'll get an error if A is not invertible, or not square.

```python
A = np.array([[1,-2,3], [0,2,2], [-4,-4,-4]])
B = np.linalg.inv(A)
B
```

| Assignment |
| :--------- |
| Execute the Python code.|
| What is the result? |
| Explain the result. |
| **Solution:** |
| <!--Insert here -->|

### Exercises

#### Matrix addition and substraction

![Matrix addition exercises](./assets/addition.png)

| Assignment |
| :--------- |
| Write the Python code for|
| A+B, B+A, A-B, B-A |
| What is your conclusion? |
| **Solution:** |
| <!--Insert here -->|

#### Scalar multiplication

![Matrix A](./assets/matrixA.png)

| Assignment |
| :--------- |
| Write the Python code for|
| 2.A |
| -3.A |
| ½.A|
| **Solution:** |
| <!--Insert here -->|

#### Matrix multiplication

![Matrix multiplication exercises](./assets/multiplication.png)

| Assignment |
| :--------- |
| Write the Python code for|
| C=A.B |
| **Solution:** |
| <!--Insert here -->|

![Matrix multiplication exercises](./assets/multiplication-3.png)

| Assignment |
| :--------- |
| Write the Python code for|
| B.C |
| A(B.C) |
| (A.B).C |
| What is your conclusion? |
| **Solution:** |
| <!--Insert here -->|

#### Translation and rotation of vectors

Consider the point P with position vector given by:

![Position vector](./assets/position.png)

In order to translate and rotate this vector it is useful to introduce an augmented vector V given by:

![Augmented vector](./assets/augmented.png)

It is then possible to define several matrices:

![Rotation x](./assets/rotx.png)

![Rotation y](./assets/roty.png)

![Rotation z](./assets/rotz.png)

![Translation](./assets/translation.png)