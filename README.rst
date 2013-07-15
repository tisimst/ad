``ad`` Package Documentation
============================

.. contents::

Overview
--------

The ``ad`` package allows you to **easily** and **transparently** perform 
**first and second-order automatic differentiation**. Advanced math 
involving trigonometric, logarithmic, hyperbolic, etc. functions can also 
be evaluated directly using the ``admath`` sub-module.

From the Wikipedia entry on `Automatic differentiation`_ (AD):

    "AD exploits the fact that every computer program, no matter how 
    complicated, executes a sequence of elementary arithmetic operations 
    (addition, subtraction, multiplication, division, etc.) and elementary 
    functions (exp, log, sin, cos, etc.). By applying the chain rule 
    repeatedly to these operations, derivatives of arbitrary order can be 
    computed automatically, and accurate to working precision."

Basic examples
--------------

Let's start with the main import::

    >>> from ad import adfloat

Creating AD objects (either a scalar or an array is acceptable)::

    >>> x = adfloat(2.0)
    >>> x
    ad(2.0)

    >>> y = adfloat([1,2,3])
    >>> y
    [ad(1.0), ad(2.0), ad(3.0)]

    >>> z = adfloat(3, tag='z')  # tags can help track variables
    >>> z
    ad(3.0, z)

Now for some math::

    >>> square = x**2
    >>> square
    ad(4.0)

    >>> sum_value = sum(y)
    >>> sum_value
    ad(6.0)

    >>> w = x*z**2
    >>> w
    ad(18.0)

Using more advanced math functions::

    >>> from ad.admath import *  # sin, cos, log, exp, sqrt, etc.
    >>> sin(1 + x**2)
    ad(-0.958924274663)

Calculating derivatives (evaluated at the given input values)::

    >>> square.d(x)  # get the first derivative wrt x
    4.0

    >>> square.d2(x)  # get the second derivative wrt x
    2.0

    >>> z.d(x)  # returns zero if the derivative doesn't exist
    0.0

    >>> w.d2c(x, z)  # second cross-derivatives, order doesn't matter
    6.0

    >>> w.d2c(z, z)  # equivalent to "w.d2(z)"
    4.0
    
    >>> w.d()  # a dict of all relevant derivatives shown if no input
    {ad(2.0): 9.0, ad(3.0, z): 12.0}

Some convenience functions (useful in optimization)::

    >>> w.gradient([x, z])  # show the gradient in the order given
    [9.0, 12.0]

    >>> w.hessian([x, z])
    [[0.0, 6.0], [6.0, 4.0]]
    
    >>> sum_value.gradient(y)  # works well with input arrays
    [1.0, 1.0, 1.0]

Working with NumPy_ arrays (most functions should work out-of-the-box)::

    >>> import numpy as np
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
-------------

- **Transparent calculations with derivatives: no or little 
  modification of existing code** is needed, including when using
  the Numpy_ module.

- **Almost all mathematical operations** are supported, including
  functions from the standard math_ module (sin, cos, exp, erf, 
  etc.) with additional convenience trigonometric, hyperbolic, 
  and logarithmic functions (csc, acoth, ln, etc.). Comparison 
  operators follow the same rules as ``float`` types.

- Nearly all derivative calculations are performed **analytically**
  (only the ``gamma`` and ``lgamma`` functions use a high-accuracy 
  finite difference formula).

Installation
------------

You have several easy, convenient options to install the ``ad`` package 
(administrative privileges may be required)

1. Download the package files below, unzip to any directory, and run 
   ``python setup.py install`` from the command-line.
   
2. Simply copy the unzipped ``ad-XYZ`` directory to any other location 
   that python can find it and rename it ``ad``.
   
3. If ``setuptools`` is installed, run ``easy_install --upgrade ad`` 
   from the command-line.
   
4. If ``pip`` is installed, run ``pip --upgrade ad`` from the command-line.

Python 3
--------

To use this package with Python 3.x, you will need to run the ``2to3`` tool at
the command-line using the following syntax while in the unzipped ``ad`` 
directory::

    $ 2to3 -w -f all *.py
    
This should take care of the main changes required. Then, run
``python3 setup.py install``. If bugs continue to pop up,
please email the author.
    
Contact
-------

Please send **feature requests, bug reports, or feedback** to 
`Abraham Lee`_.

Acknowledgements
----------------

The author expresses his thanks to `Eric O. LEBIGOT (EOL)`_, author of the uncertainties_ package, for providing code insight and inspiration.


.. _NumPy: http://numpy.scipy.org/
.. _math: http://docs.python.org/library/math.html
.. _Automatic differentiation: http://en.wikipedia.org/wiki/Automatic_differentiation
.. _Eric O. LEBIGOT (EOL): http://www.linkedin.com/pub/eric-lebigot/22/293/277
.. _uncertainties: http://pypi.python.org/pypi/uncertainties
.. _Abraham Lee: mailto:tisimst@gmail.com

