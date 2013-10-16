.. meta::
   :description: The ad Python package
   :keywords: automatic differentiation, derivative, algorithmic 
              differentiation, computational differentiation, second-order,
              optimization, Python, calculator, library, package

              
=========================
Welcome to the ad package
=========================

The `ad package`_ is a free, cross-platform python library that 
**transparently** handles calculations of first- and second-order
derivatives of nearly any mathematical expression, **regardless of the
base numeric type** (int, float, complex, etc.).

Calculations of derivatives, can be performed either in an 
**interactive session** (as with a calculator), or in **programs**
written in the Python_ programming language. Existing calculation code 
can **run with little or no change**.

.. index:: calculator

An easy-to-use calculator
=========================

Calculations involving **differentiation** can be performed 
even without knowing anything about the Python_ programming language. 
After `installing this package`_ and `invoking the Python interpreter`_, 
calculations with **automatic differentation** can be performed 
**transparently** (i.e., through the usual syntax for mathematical 
formulas):

>>> from ad import adnumber
>>> from ad.admath import *  # sin(), etc.
>>> x = adnumber(1)
>>> print 2*x
ad(2)
>>> sin(2*x)  # In a Python shell, "print" is optional
ad(0.9092974268256817)

So far, there shouldn't be anything unexpected, but first and 
second derivatives can now be accessed through **intuitive methods**:

>>> y = sin(2*x)
>>> y.d(x)  # dy/dx at x=1
-0.8322936730942848
>>> y.d2(x)  # d2y/dx2 at x=1
-3.637189707302727

Thus, existing calculation code designed for regular numbers can run 
with numbers that track derivatives with :ref:`no or little modification 
<user guide>`.

**Arrays** of numbers that track derivatives are :ref:`transparently
handled <simple_array_use>` too.

Available documentation
=======================

The :doc:`user_guide` details many of the features of this package.

Additional information is available through the pydoc_ command, which 
gives access to many of the documentation strings included in the code.

.. index:: installation

.. _installing this package:

Installation and download
=========================

Important note
--------------

The installation commands below should be **run in a DOS or Unix
command shell** (*not* in a Python shell).

Under Windows (version 7 and earlier), a command shell can be obtained
by running ``cmd.exe`` (through the Run… menu item from the Start
menu). Under Unix (Linux, Mac OS X,…), a Unix shell is available when
opening a terminal (in Mac OS X, the Terminal program is found in the
Utilities folder, which can be accessed through the Go menu in the
Finder).

Automatic install or upgrade
----------------------------

One of the automatic installation or upgrade procedures below might work 
on your system, if you have a Python package installer or use certain 
Linux distributions.

Under Unix, it may be necessary to prefix the commands below with 
``sudo``, so that the installation program has **sufficient access 
rights to the system**.

If you have `pip <http://pip.openplans.org/>`_, you can try to install
the latest version with

.. code-block:: sh

   pip install --upgrade ad

If you have setuptools_, you can try to automatically install or
upgrade this package with

.. code-block:: sh

   easy_install --upgrade ad

Manual download and install
---------------------------

Alternatively, you can simply download_ the package archive from the
Python Package Index (PyPI) and unpack it.  The package can then be
installed by **going into the unpacked directory**
(:file:`ad-...`), and running the provided :file:`setup.py`
program with

.. code-block:: sh

   python setup.py install

or, for an installation in the user Python library (no additional access
rights needed):

.. code-block:: sh

   python setup.py install --user

or, for an installation in a custom directory :file:`my_directory`:

.. code-block:: sh

   python setup.py install --install-lib my_directory

or, if additional access rights are needed (Unix):

.. code-block:: sh

   sudo python setup.py install

You can also simply **move** the :file:`ad-py*` directory
that corresponds best to your version of Python to a location that
Python can import from (directory in which scripts using
:mod:`ad` are run, etc.); the chosen
:file:`ad-py*` directory should then be renamed
:file:`ad`. Python 3 users should then run ``2to3 -w .``
from inside this directory so as to automatically adapt the code to
Python 3.

Source code
-----------

The latest, bleeding-edge but working `code
<https://github.com/tisimst/ad/tree/master/ad>`_
and `documentation source
<https://github.com/tisimst/ad/tree/master/doc/>`_ are
available `on GitHub <https://github.com/tisimst/ad/>`_.
The :mod:`ad` package is written in pure Python and has no
external dependency (the `NumPy`_ package is optional). This makes 
:mod:`ad` a **lightweight, portable package**.


Future developments
===================

Possible future developments include:

- increased support for `NumPy`_: new linear algebra methods 
  (eigenvalue and QR decompositions, determinant, etc.),
  more convenient matrix creation, etc.;

- addition of new functions from the :mod:`math` module;

- arbitrary order differentiation.

**Please support the continued development of this program** by using
`gittip <https://www.gittip.com/tisimst/>`_ by donating $10 or more.

.. index:: support

Contact
=======

**Feature requests, bug reports, or feedback are much welcome.** They
can be sent_ to the creator of :mod:`ad`, `Abraham Lee`_.

How to cite this package
========================

If you use this package for a publication (in a journal, on the web,
etc.), please cite it by including as much information as possible
from the following: *ad: a Python package for first- and 
second-order automatic differentation*, Abraham D. Lee,
`<http://pythonhosted.org/ad/>`_. Adding the version number is optional.

Acknowledgments
===============

I am greatful to Eric O. Lebigot, author of the `uncertainties package`_,
who showed me (through his code) how to easily apply the chain rule for
multivariate formulae.

I would also like to thank `Stephen Marks`_ who contributed with feedback and
suggestions on using this package with `scipy.optimize`_, which greatly 
helped improve this program.

.. index:: license

License
=======

This software is released under a **dual license**; one of the
following options can be chosen:

1. The `Revised BSD License`_ (© 2013, Abraham Lee).
2. Any other license, as long as it is obtained from the creator of
   this package.

.. _Python: http://python.org/
.. _invoking the Python interpreter: http://docs.python.org/tutorial/interpreter.html
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _download: http://pypi.python.org/pypi/ad/#downloads
.. _Eric O. LEBIGOT (EOL): http://linkedin.com/pub/eric-lebigot/22/293/277
.. _Abraham Lee: mailto:tisimst@gmail.com
.. _sent: mailto:tisimst@gmail.com
.. _Revised BSD License: http://opensource.org/licenses/BSD-3-Clause
.. _ad package: http://pypi.python.org/pypi/ad/
.. _uncertainties package: http://pypi.python.org/pypi/uncertainties
.. _pydoc: http://docs.python.org/library/pydoc.html
.. _NumPy: http://numpy.scipy.org/
.. _scipy.optimize: http://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
.. _Stephen Marks: http://economics-files.pomona.edu/marks/
