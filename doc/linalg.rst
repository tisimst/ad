.. index:: linear algebra
.. _linear algebra:

==============
Linear Algebra
==============

The :mod:`ad.linalg` submodule was created to overcome the limitations
of performing AD with compiled numerical routines (e.g., LAPACK). The 
following algorithms have a translation that are AD-compatible:

* Decompositions
  
  - ``chol``: :ref:`Cholesky decomposition <chol>`
  - ``lu``: :ref:`LU decomposition <lu>`
  - ``qr``: :ref:`QR decomposition <qr>`

* Solving linear equations and matrix inverse

  - ``solve``: :ref:`General linear system solver <solve>`
  - ``lstsq``: :ref:`Least-squares system solver <lstsq>`
  - ``inv``: :ref:`Solve for (multiplicative) matrix inverse <inv>`

Each item listed above is a NumPy-equivalent function, though not 
completely interchangeable. The descriptions that follow are not
meant to introduce theory for the methods, nor to show an exhaustive
set of examples as to their usage. They simply describe the respective
algorithm's usage with some relevant examples. Several algorithms
have been borrowed from the tasks described at 
`RosettaCode <rosettacode.org>`.

Decompositions
==============

.. index:: Decomposition; Cholesky
    
.. _chol:

Cholesky Decomposition
----------------------

`Cholesky decomposition`_ involves taking a symmetric, positive-definite 
matrix A and decomposing it into L such that :math:`A=L*L^T=U^T*U`, 
where L is a lower triangular matrix and U is an upper triangular matrix.
    
Example 1::

    >>> A = [[25, 15, -5], 
    ...      [15, 18,  0], 
    ...      [-5,  0, 11]]
    ...
    >>> L = chol(A)
    >>> L
    array([[ 5.,  0.,  0.],
           [ 3.,  3.,  0.],
           [-1.,  1.,  3.]])
    >>> U = chol(A, 'upper')
    >>> U
    array([[ 5.,  3., -1.],
           [ 0.,  3.,  1.],
           [ 0.,  0.,  3.]])
        
Example 2::

    >>> A = [[18, 22,  54,  42], 
    ...      [22, 70,  86,  62], 
    ...      [54, 86, 174, 134], 
    ...      [42, 62, 134, 106]]
    ...
    >>> L = chol(A)
    >>> L
    array([[  4.24264069,   0.        ,   0.        ,   0.        ],
           [  5.18544973,   6.5659052 ,   0.        ,   0.        ],
           [ 12.72792206,   3.0460385 ,   1.64974225,   0.        ],
           [  9.89949494,   1.62455386,   1.84971101,   1.39262125]])

.. index:: Decomposition; LU
    
.. _lu:

LU Decomposition
----------------

`LU Decomposition`_ factors a matrix as the product of a lower triangular
matrix and an upper triangular matrix, and in this case, a pivot or
permutation matrix as well. The decomposition can be viewd as the matrix
form of `guassian elimination`. Computers usually solve square systems of
linear equations using the LU decomposition, and it is also a key step
when inverting a matrix, or computing the determinant of a matrix.

Example 1::

    >>> A = [[1, 3, 5],
    ...      [2, 4, 7],
    ...      [1, 1, 0]]
    ...
    >>> L, U, P = lu(A)
    >>> L
    array([[ 1. ,  0. ,  0. ],
           [ 0.5,  1. ,  0. ],
           [ 0.5, -1. ,  1. ]])
    >>> U
    array([[ 2. ,  4. ,  7. ],
           [ 0. ,  1. ,  1.5],
           [ 0. ,  0. , -2. ]])
    >>> P
    array([[ 0.,  1.,  0.],
           [ 1.,  0.,  0.],
           [ 0.,  0.,  1.]])

Example 2::

    >>> A = [[11,  9, 24, 2], 
    ...      [ 1,  5,  2, 6], 
    ...      [ 3, 17, 18, 1], 
    ...      [ 2,  5,  7, 1]]
    ...
    >>> L, U, P = lu(A)
    >>> L
    array([[ 1.        ,  0.        ,  0.        ,  0.        ],
           [ 0.27272727,  1.        ,  0.        ,  0.        ],
           [ 0.09090909,  0.2875    ,  1.        ,  0.        ],
           [ 0.18181818,  0.23125   ,  0.00359712,  1.        ]])
    >>> U
    array([[ 11.        ,   9.        ,  24.        ,   2.        ],
           [  0.        ,  14.54545455,  11.45454545,   0.45454545],
           [  0.        ,   0.        ,  -3.475     ,   5.6875    ],
           [  0.        ,   0.        ,   0.        ,   0.51079137]])
    >>> P
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.]])

.. index:: Decomposition; QR
    
.. _qr:

QR Decomposition
----------------

`QR Decomposition`_ is applicable to any m-by-n matrix *A* and decomposes
into :math:`A=QR` where *Q* is an orthogonal matrix of size m-by-m and
*R* is an upper triangular matrix of size m-by-n. QR decomposition provides
an alternative way of `solving the systems of equations <least squares>`
:math:`Ax=b` without inverting the matrix *A*. The fact that *Q* is 
orthogonal means that :math:`Q^T*Q=I`, so that :math:`Ax=b` is equivalent to :math:`Rx=Q^T*b`, which is easier to solve since *R* is triangular.
    
Example of a square input matrix::

    >>> A = [[12, -51,   4], 
    ...      [ 6, 167, -68], 
    ...      [-4,  24, -41]]
    ...
    >>> q, r = qr(A)
    >>> q
    array([[-0.85714286,  0.39428571,  0.33142857],
           [-0.42857143, -0.90285714, -0.03428571],
           [ 0.28571429, -0.17142857,  0.94285714]])
    >>> r
    array([[ -1.40000000e+01,  -2.10000000e+01,   1.40000000e+01],
           [  5.97812398e-18,  -1.75000000e+02,   7.00000000e+01],
           [  4.47505281e-16,   0.00000000e+00,  -3.50000000e+01]])

Example of a non-square input matrix::

    >>> A = [[12, -51,   4], 
    ...      [ 6, 167, -68], 
    ...      [-4,  24, -41], 
    ...      [-1,   1,   0], 
    ...      [ 2,   0,   3]]
    ...
    >>> q, r = qr(A)
    >>> q
    array([[-0.84641474,  0.39129081, -0.34312406,  0.06613742, -0.09146206],
           [-0.42320737, -0.90408727,  0.02927016,  0.01737854, -0.04861045],
           [ 0.28213825, -0.17042055, -0.93285599, -0.02194202,  0.14371187],
           [ 0.07053456, -0.01404065,  0.00109937,  0.99740066,  0.00429488],
           [-0.14106912,  0.01665551,  0.10577161,  0.00585613,  0.98417487]])
    >>> r
    array([[ -1.41774469e+01,  -2.06666265e+01,   1.34015667e+01],
           [  3.31666807e-16,  -1.75042539e+02,   7.00803066e+01],
           [ -3.36067949e-16,   2.87087579e-15,   3.52015430e+01],
           [  9.46898347e-17,   5.05117109e-17,  -9.49761103e-17],
           [ -1.74918720e-16,  -3.80190411e-16,   8.88178420e-16]])
    >>> import numpy as np
    >>> np.all(np.dot(q, r) - A<1e-12)
    True

Solving Systems of Equations and Matrix Inverse
===============================================

.. index:: Solving linear systems; general solver
    
.. _solve:

General linear system solver
----------------------------

The general solver for linear systems of equations uses 
`guassian elimination`_. One or multiple columns in the RHS can be input,
like when solving for the `matrix inverse <inv>`.

Example::

    >>> A = [[1, 2, 1], [4, 6, 3], [9, 8, 2]]
    >>> b = [3, 2, 1]
    >>> solve(A, b)
    array([ -7.,  11., -12.])

.. index:: Solving linear systems; least-squares
    
.. _lstsq:

Least-squares linear system solver
----------------------------------

Solving a system of linear equations using the `least squares`_ method
involves the usage of `QR decomposition <least squares>`_.

Example: Fit a quadratic line to some experimental data::

    >>> x = np.array([0, 1, 2, 3, 4, 5])
    >>> y = np.array([3, 6, 11, 18, 27, 38])
    >>> y = y + np.random.randn(len(y))  # give the output a random offset
    >>> A = np.vstack([np.ones(len(x)), x, x**2]).T
    >>> A
    array([])
    >>> b = lstsq(A, y)  # the quadratic fit coefficients (b0 + b1*x + b2*x**2)
    
Now, we can see what the fit looks like compared to the original data using
Matplotlib::

    >>> fit = lambda x: b[0] + b[1]*x + b[2]*x**2
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'ro', x, fit(x), 'b-')
    >>> plt.legend(['data', 'quadratic fit'])
    >>> plt.show()

.. image:: _static/lstsq_fit.png

.. index:: Matrix inverse
    
.. _inv:

Matrix Inverse
--------------

Solving for a `matrix inverse`_ is performed using :func:`inv`.
Internally, this is done using the general solver and inputting the
an appropriately sized identity matrix as the RHS of the system.

Example::

    >>> A = [[25, 15, -5], 
    ...      [15, 18,  0], 
    ...      [-5,  0, 11]]
    ...
    >>> Ainv = inv(A)
    >>> Ainv
    array([[ 0.09777778, -0.08148148,  0.04444444],
           [-0.08148148,  0.12345679, -0.03703704],
           [ 0.04444444, -0.03703704,  0.11111111]])
    >>> np.dot(Ainv, A)
    array([[  1.00000000e+00,   0.00000000e+00,   0.00000000e+00],
           [  2.77555756e-16,   1.00000000e+00,   0.00000000e+00],
           [  0.00000000e+00,   1.11022302e-16,   1.00000000e+00]])

You'll notice that the off-diagonal elements aren't all perfectly zero. This
is due to floating-point error, but otherwise the final matrix is the 
identity matrix.

.. _Cholesky decomposition: http://en.wikipedia.org/wiki/Cholesky_decomposition
.. _LU decomposition: http://en.wikipedia.org/wiki/LU_decomposition
.. _QR decomposition: http://en.wikipedia.org/wiki/QR_decomposition
.. _gaussian elimination: http://en.wikipedia.org/wiki/Gaussian_elimination
.. _least squares: http://en.wikipedia.org/wiki/Linear_least_squares_(mathematics)#Orthogonal_decomposition_methods
.. _matrix inverse: http://en.wikipedia.org/wiki/Gaussian_elimination#Finding_the_inverse_of_a_matrix

Basic setup
===========

Basic mathematical operations involving numbers that track derivatives
only require a simple import:

>>> from ad import adnumber

The :func:`adnumber` function creates numbers with derivative tracing
capabilities. Existing calculation code can usually run with no or 
little modification and automatically produce derivatives.

.. The "import ad" is put here because some examples requires
   ad to have been imported (and not only adnumber).

The :mod:`ad` module contains other features, which can be
made accessible through

>>> import ad

The :mod:`ad` package also contains sub-modules for
:ref:`advanced mathematical functions <advanced math operations>`

.. index::
   pair: number that tracks derivatives; creation

Creating automatical differentiation numbers
============================================

Numbers that track their derivatives are input just as you would for
any normal numeric type. In that sense, they are basically *wrapped*
without really changing their fundamental type. For example, x = 2
can be expressed in any of the basic numeric ways, including:

>>> x = adnumber(2)  # acts like an "int" object
>>> x = adnumber(2.0)  # acts like a "float" object
>>> x = adnumber(2+0j)  # acts like a "complex" object

Mathematical calculations that follow, like:

>>> x/3  # if x = adnumber(2), the result is ad(0) because of integer math

are interpreted based upon the base numeric types involved.

Basic math
==========

Calculations can be performed directly, as with regular real or complex
numbers:

>>> square = x**2
>>> print square
ad(4)
>>> a = adnumber(3 + 4j)
>>> print a
ad((3+4j))
>>> abs(a)
ad(5.0)
>>> b = adnumber(1 - 1j)
>>> a*b
ad((7+1j))
>>> a.real, a.imag
(3.0, 4.0)

AD objects that represent real values can also be used to create complex ones:

>>> y = adnumber(3.14)
>>> z = x + y*1j
>>> print z
ad((2+3.14j))

If an AD object is used as input to :func:`adnumber`, then a deepcopy is made,
but no tracking relation is created between the input and output objects:

>>> z = adnumber(x)
>>> z
ad(2)
>>> z is x  # are they the same object?
False
>>> z == x  # are the nominal values the same?
True

More of this is explained below, but we can see that derivatives show the
disconnect between the objects:

>>> z.d(x)  # derivative of z wrt x
0.0
>>> z.d(z)  # derivative of z wrt itself
1.0
>>> x.d(z)  # derivative of x wrt z
0.0


.. index:: mathematical operation; on a scalar, admath

.. _advanced math operations:

Mathematical operations
=======================

Besides being able to apply basic mathematical operations, this package 
provides generalizations of **most of the functions from the standard** 
:mod:`math` **and** :mod:`cmath` **modules**.  
These mathematical functions are found in the :mod:`ad.admath` module:

>>> from ad.admath import *  # Imports sin(), etc.
>>> sin(x**2)
ad(-0.7568024953079282)

These functions are designed to support whichever numeric types are normally
compatible with them. For example, the :func:`sin` function has a real and
complex counterpart, as do many others. Some functions, like :func:`erf` are 
only available in the :mod:`math` module, so an exception is raised if a
complex number is passed to it.

There are also many other functions not normally found in the :mod:`math`
and :mod:`cmath` modules that are conveniently available, like :func:`csc`
and others.

The list of available mathematical functions can be obtained with the
``pydoc ad.admath`` command.


.. index:: arrays; simple use, matrices; simple use

.. _simple_array_use:

Arrays of numbers
=================

It is possible to put automatic differentiation numbers with in NumPy_ 
arrays and matrices, lists, or tuples, and the returned object is of that
respective type (even nested objects work):

>>> adnumber([1, [2, 3]])  # nested list input
[ad(1), [ad(2), ad(3)]]
>>> adnumber((1, 2))  # tuple input
(ad(1), ad(2))
>>> arr = adnumber(np.array([[1, 2], [3, 4]]))  # NumPy array input
>>> 2*arr
array([[ad(2), ad(4)],
       [ad(6), ad(8)]], dtype=object)
>>> print arr.sum()
ad(10)

Thus, usual operations on NumPy arrays can be performed transparently
even when these arrays contain numbers that track derivatives.


.. index::
   pair: nominal value; of scalar
   pair: derivative; of scalar

Access to the derivatives and to the nominal value
==================================================

The nominal value and the derivatives can be accessed independently:

>>> print square
ad(4)
>>> print square.x  # the underlying numeric object
4
>>> print square.d(x)  # first derivative wrt x
4.0
>>> print square.d2(x)  # second derivative wrt x
2.0
>>> print square.d()  # left empty returns a dictionary of related derivatives
{ad(4): 4.0}
>>> y = adnumber(1.5)
>>> print square.d(y)  # if a derivative doesn't exist, zero is returned
0.0
>>> z = square/y
>>> z.d2c(x,y)  # second cross-derivative wrt x and y, either order is ok
-1.7777777777777777
>>> z.d(square)  # since "square" is a dependent variable, derivative is zero
0.0

Access to more than one derivative
==================================

Arrays of derivatives can be obtained through the :func:`gradient` and 
:func:`hessian` methods. The individual variables can be more easily
identified when the variables are **tagged**, though tags *do not have
to be distinct*:

>>> u = adnumber(0.1, 'u')  # Tag
>>> v = adnumber(3.14, 'v')

>>> sum_value = u+2*v/u
>>> sum_value
ad(62.9)
>>> sum_value.d()
{ad(0.1, u): -626.9999999999999, ad(3.14, v): 20.0}

>>> sum_value.gradient([u, v])
[-626.9999999999999, 20.0]

>>> sum_value.hessian([u, v])
[[12559.999999999998, -199.99999999999997], [-199.99999999999997, 0.0]]

The **jacobian matrix** can be easily created for multiple dependent objects,
where each row is the gradient of the dependent variables with respect to
each of the independent variables, *in the order specified*:

>>> from ad import jacobian
>>> jacobian([square, sum_value], [x, u, v])
[[4.0, 0.0, 0.0], [0.0, -626.9999999999999, 20.0]]


.. index:: comparison operators

Comparison operators
====================

Comparison operators behave naturally as they would with numbers outside of 
this package, even with other scalar values:

>>> x = adnumber(0.2)
>>> x
ad(0.2)
>>> y = adnumber(1)
>>> y
ad(1)
>>> y > x
True
>>> y > 0
True
>>> y == 1.0
True


.. index::
   single: C code; wrapping
   single: Fortran code; wrapping
   single: wrapping (C, Fortran,…) functions

Making custom functions accept numbers that track derivatives
=============================================================

Due to the nature of automatic differentiation, unless a function can be
represented with a mathematical equation, automatic differentiation is
meaningless. For custom functions that cannot be represented mathematically
(i.e., those that do not have an analytical form), derivatives may be 
calculated using other means like finite difference derivatives.


Miscellaneous utilities
=======================

.. index:: optimization

It is sometimes useful to use the gradients and hessians provided by this
package for the purpose of supplementing an optimization routine, like
those in the `scipy.optimize`_ submodule. With this package, a function
can be conveniently wrapped with functions that return both the gradient
and hessian:

>>> from ad import gh  # the gradient and hessian functions generator
>>> def my_cool_function(x):
...     return (x[0] - 10.0)**2 + (x[1] + 5.0)**2
>>> my_cool_gradient, my_cool_hessian = gh(my_cool_function)

These objects (:func:`my_cool_gradient` and :func:`my_cool_hessian`) 
*are functions* that accept an array ``x`` and other optional args. 
Depending on the optimization routine, you may be able to use only
the gradient function (typically with the keyword ``jac=...``):

>>> from scipy.optimize import minimize
>>> x0 = [24, 17]
>>> bnds = ((0, None), (0, None))
>>> res = minimize(my_cool_function, x0, bounds=bnds, method='L-BFGS-B', 
...     jac=my_cool_gradient, options={'ftol': 1e-8, 'disp':False})
>>> res.x
array([ 10.,   0.])
>>> res.fun
25.0
>>> res.jac
array([  7.10542736e-15,   1.00000000e+01])

You might wonder why the final gradient (``res.jac``) isn't precisely [0, 10].
It's *not* because of numerical error in the AD methods, because although it
appears that the final ``res.x`` value is precisely [10, 0], if we print out 
all the available digits, we see that this is not the case (i.e., NumPy_ 
was not being completely truthful about it's values with pretty printing):

>>> list(res.x)
[10.000000000000004, 0.0]

The real reason for the inaccuracy is because of the ``None`` upper bounds
given to the `L-BFGS-B` algorithm. If we had used finite upper bounds, 
we get the exact answer:

>>> bnds = ((0, 100), (0, 100))
>>> res = minimize(my_cool_function, x0, bounds=bnds, method='L-BFGS-B', 
...     jac=my_cool_gradient, options={'ftol': 1e-8, 'disp':True})
>>> list(res.x)
[10.0, 0.0]
>>> list(res.jac)
[0.0, 10.0]

Notice that the use of :func:`gh` doesn't require you to explicitly 
initialize any variable with :func:`adnumber` since it all happens 
internally with the wrapped functions.


Testing if a variable can track derivatives
===========================================

The recommended way of testing whether :data:`value` tracks
derivatives handled by this module is by checking whether
:data:`value` is an instance of :class:`ADF`, through
``isinstance(value, ad.ADF)``.


Python classes for variables and functions with derivatives
===========================================================

Numbers with derivatives are represented through two different
classes:

1. a class for independent variables (:class:`ADV`, which
   inherits from :class:`ADF`),

2. a class for functions that depend on independent variables
   (:class:`ADF`).

Documentation for these classes is available in their Python
docstring, which can for instance displayed through pydoc_.

The factory function :func:`adnumber` creates variables and thus returns
a :class:`ADV` object:

>>> x = adnumber(0.1)
>>> type(x)
<class 'ad.ADV'>

:class:`ADV` objects can be used as if they were regular Python
numbers (the summation, etc. of these objects is defined).

Mathematical expressions involving numbers with derivatives
generally return :class:`ADF` objects, because they
represent mathematical functions and not simple variables; these
objects store all the variables they depend on:

>>> type(admath.sin(x))
<class 'ad.ADF'>

.. _NumPy: http://numpy.scipy.org/
.. _scipy.optimize: http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
.. _pydoc: http://docs.python.org/library/pydoc.html

