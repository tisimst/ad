.. index:: user guide
.. _user guide:

==========
User Guide
==========


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

