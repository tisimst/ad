Overview
========
The ``ad`` package allows you to **easily** and **transparently** perform 
**first and second-order automatic differentiation**. Advanced math 
involving trigonometric, logarithmic, hyperbolic, etc. functions can also 
be evaluated directly using the ``admath`` sub-module.

`Automatic differentiation`_ is different from numerical and symbolic
differentiation in that it uses prior knowledge of how derivatives 
are calculated, `but that's the part you don't need to worry about 
while using this package`. They are then transmitted through subsequent 
calculations (using the generalized `chain rule`_).

Basic examples
==============
::

    >>> from ad import AD

    >>> x = adfloat(2.0)
    >>> x
    ad(2.0)

    >>> square = x**2
    >>> square
    ad(4.0)
    >>> square.d(x)  # get the first derivative wrt x
    4.0
    >>> square.d2(x)  # get the second derivative wrt x
    2.0

    >>> from ad.admath import *  # sin, cos, log, exp, sqrt, etc.
    >>> sin(1 + x**2)
    ad(-0.958924274663)

    >>> print (2*x + 1000).d()  # no inputs shows dict of all derivatives
    {ad(2.0): 2.0}

    >>> y = adfloat(3, tag='y')  # tags are useful for tracking original variables
    >>> y
    ad(3.0)
    >>> y.d(x)  # returns zero if the derivative doesn't exist
    0.0

    >>> z = x*y**2
    >>> z
    ad(18.0)
    >>> z.gradient([x, y])  # show the gradient in the order given
    [9.0, 12.0]
    >>> z.d2c(x, y)  # second cross-derivatives, order doesn't matter -> (x,y) or (y,x)
    6.0
    >>> z.hessian([x, y])
    [[0.0, 6.0], [6.0, 4.0]]

    >>> import numpy as np  # most numpy functions work out of the box
    >>> arr = np.array(adfloat([1, 2, 3]))  # multiple input support
    >>> arr.sum()
    ad(6.0)
    >>> arr.max()
    ad(3.0)
    >>> arr.mean()
    ad(2.0)
    >>> arr.var()  # array variance
    ad(0.666666666667)
    >>> sqrt(arr)  # vectorized operations supported with ad operators
    array([ad(1.0), ad(1.41421356237), ad(1.73205080757)], dtype=object)

Main Features
=============

- **Transparent calculations with derivatives: no or little 
  modification of existing code** is needed, including when using
  the `Numpy`_ module. The only function (that I have tested, and I 
  certainly haven't tested most of them) that doesn't work right 
  out of the box is ``numpy.std`` since it internally calls its 
  builtin ``sqrt`` function. Two alternatives exist to work around 
  this: 1) use ``**0.5`` or 2) using the ``admath.sqrt`` 
  operator.
- **Almost all mathematical operations** are supported, including
  functions from the standard `math`_ module (sin, cos, exp, erf, 
  etc.) with additional convenience trigonometric, hyperbolic, 
  and logarithmic functions (csc, acoth, ln, etc.). Comparison 
  operators follow the same rules as ``float`` types.
- Nearly all derivative calculations are performed **analytically**
  (only the ``gamma`` and ``lgamma`` functions use a high-accuracy 
  finite difference formula).

Installation
============

You have several easy, convenient options to install the ``ad`` package 
(administrative privileges may be required)

1. Download the package files below, unzip to any directory, and run 
   ``python setup.py install`` from the command-line
2. Simply copy the unzipped ``ad-XYZ`` directory to any other location 
   that python can find it and rename it ``ad``
3. If ``setuptools`` is installed, run ``easy_install --upgrade ad`` 
   from the command-line
4. If ``pip`` is installed, run ``pip --upgrade ad`` from the command-line

Contact
=======

Please send **feature requests, bug reports, or feedback** to 
`Abraham Lee`_.


.. _NumPy: http://numpy.scipy.org/
.. _math: http://docs.python.org/library/math.html
.. _Automatic differentiation: http://en.wikipedia.org/wiki/Automatic_differentiation
.. _chain rule: http://en.wikipedia.org/wiki/Chain_rule
.. _EOL: http://www.linkedin.com/pub/eric-lebigot/22/293/277
.. _Abraham Lee: mailto:tisimst@gmail.com

