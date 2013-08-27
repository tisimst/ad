# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:52:09 2013

@author: tisimst
"""
import math
import cmath
import copy
from random import randint

try:
    import numpy
    numpy_installed = True
except ImportError:
    numpy_installed = False

__version_info__ = (1, 1, 2)
__version__ = '.'.join(map(str, __version_info__))

__author__ = 'Abraham Lee'

__all__ = ['adnumber', 'gh']

CONSTANT_TYPES = (float, int, long, complex)

def to_auto_diff(x):
    """
    Transforms x into a automatically differentiated function (ADF),
    unless it is already an ADF (or a subclass of it), in which case x is 
    returned unchanged.

    Raises an exception unless 'x' belongs to some specific classes of
    objects that are known not to depend on ADF objects (which then cannot be
    considered as constants).
    """

    if isinstance(x, ADF):
        return x

    #! In Python 2.6+, numbers.Number could be used instead, here:
    if isinstance(x, CONSTANT_TYPES):
        # constants have no derivatives to define:
        return ADF(x, {}, {}, {})

    raise NotImplementedError(
        'Automatic differentiation not yet supported for {:} objects'.format(
        type(x))
        )
        
#def partial_derivative(f, param_num):
#    """
#    Returns a function that numerically calculates the partial
#    derivative of function f with respect to its argument number
#    param_num.
#    """
#
#    def partial_derivative_of_f(*args):
#        """
#        Partial derivative, calculated with the (-epsilon, +epsilon)
#        method, which is more precise than the (0, +epsilon) method.
#        """
#        # f_nominal_value = f(*args)
#
#        shifted_args = list(args)  # Copy, and conversion to a mutable
#
#        # The step is relative to the parameter being varied, so that
#        # shifting it does not suffer from finite precision:
#        step = 1e-8*abs(shifted_args[param_num])
#        if not step:
#            # Arbitrary, but "small" with respect to 1, and of the
#            # order of the square root of the precision of double
#            # precision floats:
#            step = 1e-8
#
#        shifted_args[param_num] += step
#        shifted_f_plus = f(*shifted_args)
#        
#        shifted_args[param_num] -= 2*step  # Optimization: only 1 list copy
#        shifted_f_minus = f(*shifted_args)
#
#        return (shifted_f_plus - shifted_f_minus)/2/step
#
#    return partial_derivative_of_f
#
#def second_partial_derivative(f, param_num):
#    """
#    Returns a function that numerically calculates the second partial
#    derivative of function f with respect to its argument number.
#    """
#    def second_partial_derivative_of_f(*args):
#        """
#        Partial derivative, calculated with the (-epsilon, +epsilon)
#        method, which is more precise than the (0, +epsilon) method.
#        """
#        f_nominal_value = f(*args)
#
#        shifted_args = list(args)  # Copy, and conversion to a mutable
#
#        # The step is relative to the parameter being varied, so that
#        # shifting it does not suffer from finite precision:
#        step = 1e-8*abs(shifted_args[param_num])
#        if not step:
#            # Arbitrary, but "small" with respect to 1, and of the
#            # order of the square root of the precision of double
#            # precision floats:
#            step = 1e-8
#
#        shifted_args[param_num] += step
#        shifted_f_plus = f(*shifted_args)
#        
#        shifted_args[param_num] -= 2*step  # Optimization: only 1 list copy
#        shifted_f_minus = f(*shifted_args)
#
#        return (shifted_f_plus - 2*f_nominal_value + shifted_f_minus)/step**2
#
#    return second_partial_derivative_of_f
#
#def cross_partial_derivative(f, param_num1, param_num2):
#    """
#    Returns a function that numerically calculates the cross-second partial
#    derivative of function f with respect to its two argument numbers.
#    """
#    if param_num1==param_num2:
#        return second_partial_derivative(f,param_num1)
#        
#    def cross_partial_derivative_of_f(*args):
#        """
#        Partial derivative, calculated with the (-epsilon, +epsilon)
#        method, which is more precise than the (0, +epsilon) method.
#        """
##        f_nominal_value = f(*args)
#
#        shifted_args = list(args)  # Copy, and conversion to a mutable
#
#        # The step is relative to the parameter being varied, so that
#        # shifting it does not suffer from finite precision:
#        step1 = 1e-8*abs(shifted_args[param_num1])
#        step2 = 1e-8*abs(shifted_args[param_num2])
#        if not step1:
#            # Arbitrary, but "small" with respect to 1, and of the
#            # order of the square root of the precision of double
#            # precision floats:
#            step1 = 1e-8
#        if not step2:
#            # Arbitrary, but "small" with respect to 1, and of the
#            # order of the square root of the precision of double
#            # precision floats:
#            step2 = 1e-8
#
#        shifted_args[param_num1] += step1
#        shifted_args[param_num2] += step2
#        shifted_f_plus_plus = f(*shifted_args)
#        
#        shifted_args[param_num1] -= 2*step1  # Optimization: only 1 list copy
#        shifted_f_minus_plus = f(*shifted_args)
#
#        shifted_args[param_num1] += 2*step1
#        shifted_args[param_num2] -= 2*step2
#        shifted_f_plus_minus = f(*shifted_args)
#        
#        shifted_args[param_num1] -= 2*step1  # Optimization: only 1 list copy
#        shifted_f_minus_minus = f(*shifted_args)
#
#        return ((shifted_f_plus_plus - shifted_f_minus_plus)-\
#                (shifted_f_plus_minus - shifted_f_minus_minus))/4/step1/step2
#
#    return cross_partial_derivative_of_f
#
#class NumericalDerivatives(object):
#    """
#    Convenient access to the partial derivatives of a function,
#    calculated numerically.
#    """
#    # This is not a list because the number of arguments of the
#    # function is not known in advance, in general.
#
#    def __init__(self, function, order):
#        """
#        'function' is the function whose derivatives can be computed.
#        """
#        self._function = function
#        self._order = order
#
#    def __getitem__(self, n, order=1):
#        """
#        Returns the n-th numerical derivative of the function.
#        """
#        if order==1:
#            return partial_derivative(self._function, n)
#        elif order==2:
#            if hasattr(n,'__iter__'):
#                return cross_partial_derivative(self._function,n[0],n[1])
#            else:
#                return second_partial_derivative(self._function,n)
#
#
#def wrap(f, deriv_wrt_args=None, deriv2_wrt_args=None, deriv2c_wrt_args=None):
#    """
#    Wraps an arbitrary function f so that it can accept ADV/ADF objects and 
#    return an ADF object that contains first and second partial derivatives 
#    wrt ADV objects.
#
#    Parameters
#    ----------
#    f : function
#        Any function that returns a scalar (not a list or array-like object)
#
#    Optional
#    --------
#    deriv_wrt_args : list
#        1st derivatives of f with respect to its input arguments (pure linear
#        terms)
#    deriv2_wrt_args : list
#        2nd derivatives of f with respect to its input arguments (pure 
#        quadratic terms)
#    deriv2c_wrt_args : list
#        2nd cross-product derivatives of f with respect to its input arguments
#        (i.e., if f(x,y), then this only contains d^2f/dxdy). This list should
#        contain the upper triangle of the hessian matrix (not including the
#        diagonal, pure quadratic terms), with entries arranged row by row.
#        
#        Thus, if the hessian of a function of three variables is:
#            
#                    | . d^2f/dx1dx2  d^2f/dx1dx3|
#             H(f) = | .      .       d^2f/dx2dx3|
#                    | .      .            .     |
#        
#        then deriv2c_wrt_args = [d^2f/dx1dx2, d^2f/dx1dx3, d^2f/dx2dx3]
#        
#    """
#
#    if deriv_wrt_args is None:
#        deriv_wrt_args = NumericalDerivatives(f, 1)
#    else:
#        # Derivatives that are not defined are calculated numerically,
#        # if there is a finite number of them (the function lambda
#        # *args: fsum(args) has a non-defined number of arguments, as
#        # it just performs a sum):
#        try:  # Is the number of derivatives fixed?
#            len(deriv_wrt_args)
#        except TypeError:
#            pass
#        else:
#            deriv_wrt_args = [
#                partial_derivative(f, k) if derivative is None
#                else derivative
#                for (k, derivative) in enumerate(deriv_wrt_args)]
#
#    if deriv2_wrt_args is None:
#        deriv2_wrt_args = NumericalDerivatives(f, 2)
#    else:
#        # Derivatives that are not defined are calculated numerically,
#        # if there is a finite number of them (the function lambda
#        # *args: fsum(args) has a non-defined number of arguments, as
#        # it just performs a sum):
#        try:  # Is the number of derivatives fixed?
#            len(deriv2_wrt_args)
#        except TypeError:
#            pass
#        else:
#            deriv2_wrt_args = [
#                second_partial_derivative(f, k) if derivative is None
#                else derivative
#                for (k, derivative) in enumerate(deriv2_wrt_args)]
#
#    if deriv2c_wrt_args is None:
#        deriv2c_wrt_args = NumericalDerivatives(f, 2)
#    else:
#        try:
#            len(deriv2c_wrt_args)
#        except TypeError:
#            pass
#        else:
#            deriv2c_wrt_args = [[
#                cross_partial_derivative(f,k1,k2) if derivative is None and k1<k2
#                else derivative
#                for (k1, derivative) in enumerate(deriv2c_wrt_args)]
#                for (k2, derivative) in enumerate(deriv2c_wrt_args)]
#
#
#
#    # THE USUAL CODE BELOW NEEDS TO BE MODIFIED ###############################
#
#    #    ad_funcs = map(to_auto_diff,(self,val))
#    #
#    #    x = ad_funcs[0].x
#    #    y = ad_funcs[1].x
#    #    
#    #    ########################################
#    #    # Nominal value of the constructed ADF:
#    #    f_nominal   = x + y
#    #    
#    #    ########################################
#    #    variables = self._get_variables(ad_funcs)
#    #    
#    #    if not variables or isinstance(f_nominal, bool):
#    #        return f
#    #
#    #    ########################################
#    #
#    #    # Calculation of the derivatives with respect to the arguments
#    #    # of f (ad_funcs):
#    #
#    #    lc_wrt_args = [1., 1.]
#    #    qc_wrt_args = [0., 0.]
#    #    cp_wrt_args = 0.
#    #
#    #    ########################################
#    #    # Calculation of the derivative of f with respect to all the
#    #    # variables (Variable) involved.
#    #
#    #    lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
#    #                                ad_funcs,variables,lc_wrt_args,qc_wrt_args,
#    #                                cp_wrt_args)
#    #                                
#    #    # The function now returns an ADF object:
#    #    return ADF(f_nominal, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)

def _apply_chain_rule(ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                           cp_wrt_args):
    """
    This function applies the first and second-order chain rule to calculate the
    derivatives with respect to original variables (i.e., objects created with
    the ``adnumber(...)`` constructor).
    
    For reference:
    - ``lc`` refers to "linear coefficients" or first-order terms
    - ``qc`` refers to "quadratic coefficients" or pure second-order terms
    - ``cp`` refers to "cross-product" second-order terms
    
    """
    # Initial value (is updated below):
    lc_wrt_vars = dict((var, 0.) for var in variables)
    qc_wrt_vars = dict((var, 0.) for var in variables)
    cp_wrt_vars = {}
    for i,var1 in enumerate(variables):
        for j,var2 in enumerate(variables):
            if i<j:
                cp_wrt_vars[(var1,var2)] = 0.

    # The chain rule is used (we already have
    # derivatives_wrt_args):
    for j,var1 in enumerate(variables):
        for k,var2 in enumerate(variables):
            for (f, dh, d2h) in zip(ad_funcs,lc_wrt_args,qc_wrt_args):
                
                if j==k:
                    # first order terms
                    tmp = dh*f.d(var1)
                    lc_wrt_vars[var1] += tmp

                    # pure second-order terms
                    tmp = dh*f.d2(var1) + d2h*f.d(var1)**2
                    qc_wrt_vars[var1] += tmp

                elif j<k:
                    # cross-product second-order terms
                    tmp = dh*f.d2c(var1,var2)
                    cp_wrt_vars[(var1,var2)] += tmp
                    tmp = d2h*f.d(var1)*f.d(var2)
                    cp_wrt_vars[(var1,var2)] += tmp

            # now add in the other cross-product contributions to second-order
            # terms
            if j==k and len(ad_funcs)>1:
                tmp = 2*cp_wrt_args*ad_funcs[0].d(var1)*ad_funcs[1].d(var1)
                qc_wrt_vars[var1] += tmp

            elif j<k and len(ad_funcs)>1:
                tmp = cp_wrt_args*(ad_funcs[0].d(var1)*ad_funcs[1].d(var2) + \
                                   ad_funcs[0].d(var2)*ad_funcs[1].d(var1))
                cp_wrt_vars[(var1,var2)] += tmp
                
    return (lc_wrt_vars,qc_wrt_vars,cp_wrt_vars)
    
def _floor(x):
    """
    Return the floor of x as a float, the largest integer value less than or 
    equal to x. This is required for the "mod" function.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = _floor(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [0.0]
        qc_wrt_args = [0.0]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        return math.floor(x)

class ADF(object):
    """
    The ADF (Automatically Differentiated Function) class contains derivative
    information about the results of a previous operation on any two objects 
    where at least one is an ADF or ADV object. 
    
    An ADF object has class members '_lc', '_qc', and '_cp' to contain 
    first-order derivatives, second-order derivatives, and cross-product 
    derivatives, respectively, of all ADV objects in the ADF's lineage. When 
    requesting a cross-product term, either order of objects may be used since, 
    mathematically, they are equivalent. 
    
    For example, if z = z(x, y), then::

          2       2
         d z     d z
        ----- = -----
        dx dy   dy dx
    
    
    Example
    -------
    Initialize some ADV objects (tag not required, but useful)::

        >>> x = adnumber(1, tag='x')
        >>> y = adnumber(2, tag='y')
        
    Now some basic math, showing the derivatives of the final result. Note that
    if we don't supply an input to the derivative methods, a dictionary with
    all derivatives wrt the subsequently used ADV objects is returned::
        
        >>> z = x + y
        >>> z.d()
        {ad(1.0, x): 1.0, ad(2.0, y): 1.0}
        >>> z.d2()
        {ad(1.0, x): 0.0, ad(2.0, y): 0.0}
        >>> z.d2c()
        {(ad(1.0, x), ad(2.0, y)): 0.0}
        
    Let's take it a step further now and see if relationships hold::
        
        >>> w = x*z  # same as x*(x+y) = x**2 + x*y
        >>> w.d(x)  # dw/dx = 2*x+y = 2*(1) + (2) = 4
        4.0
        >>> w.d2(x)  # d2w/dx2 = 2
        2.0
        >>> w.d2(y)  # d2w/dy2 = 0
        0.0
        >>> w.d2c(x, y)  # d2w/dxdy = 1
        1.0

    For convenience, we can get the gradient and hessian if we supply the order
    of the variables (useful in optimization routines)::
        
        >>> w.gradient([x, y])
        [4.0, 1.0]
        >>> w.hessian([x, y])
        [[2.0, 1.0], [1.0, 0.0]]
        
    You'll note that these are constructed using lists and nested lists instead
    of depending on numpy arrays, though if numpy is installed, they can look
    much nicer and are a little easier to work with::
        
        >>> import numpy as np
        >>> np.array(w.hessian([x, y]))
        array([[ 2.,  1.],
               [ 1.,  0.]])

    """
    __slots__ = ['x', '_lc', '_qc', '_cp', 'tag', '_trace']
    
    def __init__(self, value, lc, qc, cp, tag=None):
        # I want to be able to perform complex derivatives, so "x" will
        # assume whatever type of object is put into it.
        self.x = value
        self._lc = lc
        self._qc = qc
        self._cp = cp
        self.tag = tag
        self._trace = None
    
    def __hash__(self):
      return id(self)
    
    @property
    def real(self):
        return self.x.real
    
    @property
    def imag(self):
        return self.x.imag
    
    def _to_general_representation(self, str_func):
        """
        This provides the general representation of the underlying numeric 
        object, but assumes self.tag is a string object.
        """
        if self.tag is None:
            return 'ad({:})'.format(str_func(self.x))
        else:
            return 'ad({:}, {:})'.format(str_func(self.x), str(self.tag))
        
    def __repr__(self):
        return self._to_general_representation(repr)
    
    def __str__(self):
        return self._to_general_representation(str)

    def d(self, x=None):
        """
        Returns first derivative with respect to x (an AD object).
        
        Optional
        --------
        x : AD object
            Technically this can be any object, but to make it practically 
            useful, ``x`` should be a single object created using the 
            ``adnumber(...)`` constructor. If ``x=None``, then all associated 
            first derivatives are returned in the form of a ``dict`` object.
                    
        Returns
        -------
        df/dx : scalar
            The derivative (if it exists), otherwise, zero.
            
        Examples
        --------
        ::
            >>> x = adnumber(2)
            >>> y = 3
            >>> z = x**y
            
            >>> z.d()
            {ad(2): 12.0}
            
            >>> z.d(x)
            12.0
            
            >>> z.d(y)  # derivative wrt y is zero since it's not an AD object
            0.0
            
        See Also
        --------
        d2, d2c, gradient, hessian
        
        """
        if x is not None:
            if isinstance(x, ADF):
                try:
                    tmp = self._lc[x]
                except KeyError:
                    tmp = 0.0
                return tmp if tmp.imag else tmp.real
            else:
                return 0.0
        else:
            return self._lc
    
    def d2(self, x=None):
        """
        Returns pure second derivative with respect to x (an AD object).
        
        Optional
        --------
        x : AD object
            Technically this can be any object, but to make it practically 
            useful, ``x`` should be a single object created using the 
            ``adnumber(...)`` constructor. If ``x=None``, then all associated 
            second derivatives are returned in the form of a ``dict`` object.
                    
        Returns
        -------
        d2f/dx2 : scalar
            The pure second derivative (if it exists), otherwise, zero.
            
        Examples
        --------
        ::
            >>> x = adnumber(2.5)
            >>> y = 3
            >>> z = x**y
            
            >>> z.d2()
            {ad(2): 15.0}
            
            >>> z.d2(x)
            15.0
            
            >>> z.d2(y)  # second deriv wrt y is zero since not an AD object
            0.0
            
        See Also
        --------
        d, d2c, gradient, hessian
        
        """
        if x is not None:
            if isinstance(x, ADF):
                try:
                    tmp = self._qc[x]
                except KeyError:
                    tmp = 0.0
                return tmp if tmp.imag else tmp.real
            else:
                return 0.0
        else:
            return self._qc
    
    def d2c(self, x=None, y=None):
        """
        Returns cross-product second derivative with respect to two objects, x
        and y (preferrably AD objects). If both inputs are ``None``, then a dict
        containing all cross-product second derivatives is returned. This is 
        one-way only (i.e., if f = f(x, y) then **either** d2f/dxdy or d2f/dydx
        will be in that dictionary and NOT BOTH). 
        
        If only one of the inputs is ``None`` or if the cross-product 
        derivative doesn't exist, then zero is returned.
        
        If x and y are the same object, then the pure second-order derivative
        is returned.
        
        Optional
        --------
        x : AD object
            Technically this can be any object, but to make it practically 
            useful, ``x`` should be a single object created using the 
            ``adnumber(...)`` constructor.
        y : AD object
            Same as ``x``.
                    
        Returns
        -------
        d2f/dxdy : scalar
            The pure second derivative (if it exists), otherwise, zero.
            
        Examples
        --------
        ::
            >>> x = adnumber(2.5)
            >>> y = adnumber(3)
            >>> z = x**y
            
            >>> z.d2c()
            {(ad(2.5), ad(3)): 33.06704268553368}
            
            >>> z.d2c(x, y)  # either input order gives same result
            33.06704268553368
            
            >>> z.d2c(y, y)  # pure second deriv wrt y
            0.8395887053184748
            
        See Also
        --------
        d, d2, gradient, hessian
        
        """
        if (x is not None) and (y is not None):
            if x is y:
                tmp = self.d2(x)
            else:
                if isinstance(x, ADF) and isinstance(y, ADF):
                    try:
                        tmp = self._cp[(x, y)]
                    except KeyError:
                        try:
                            tmp = self._cp[(y, x)]
                        except KeyError:
                            tmp = 0.0
                else:
                    tmp = 0.0
                
            return tmp if tmp.imag else tmp.real

        elif ((x is not None) and not (y is not None)) or \
             ((y is not None) and not (x is not None)):
            return 0.0
        else:
            return self._cp
    
    def gradient(self, variables):
        """
        Returns the gradient, or Jacobian, (array of partial derivatives) of the
        AD object given some input variables. The order of the inputs
        determines the order of the returned list of values::
        
            f.gradient([y, x, z]) --> [df/dy, df/dx, df/dz]
        
        Parameters
        ----------
        variables : array-like
            An array of objects (they don't have to be AD objects). If a partial
            derivative doesn't exist, then zero will be returned. If a single
            object is input, a single derivative will be returned as a list.
            
        Returns
        -------
        grad : list
            An list of partial derivatives
            
        Example
        -------
        ::
            >>> x = adnumber(2)
            >>> y = adnumber(0.5)
            >>> z = x**y
            >>> z.gradient([x, y])
            [0.3535533905932738, 0.9802581434685472]

            >>> z.gradient([x, 3, 0.4, y, -19])
            [0.9802581434685472, 0.0, 0.0, 0.3535533905932738, 0.0]
            
        See Also
        --------
        hessian, d, d2, d2c
        """
        try:
            grad = [self.d(v) for v in variables]
        except TypeError:
            grad = [self.d(variables)]
        return grad
        
    def hessian(self, variables):
        """
        Returns the hessian (2-d array of second partial derivatives) of the AD 
        object given some input variables. The output order is determined by the
        input order::
        
            f.hessian([y, x, z]) --> [[d2f/dy2, d2f/dydx, d2f/dydz],
                                      [d2f/dxdy, d2f/dx2, d2f/dxdz],
                                      [d2f/dzdy, d2f/dzdx, d2f/dz2]]
        
        Parameters
        ----------
        variables : array-like
            An array of objects (they don't have to be AD objects). If a partial
            derivative doesn't exist, the result of that item is zero as 
            expected. If a single object is input, a single second derivative 
            will be returned as a nested list.
            
        Returns
        -------
        hess : 2d-list
            An nested list of second partial derivatives (pure and 
            cross-product)
            
        Example
        -------
        ::
            >>> x = adnumber(2)
            >>> y = adnumber(0.5)
            >>> z = x**y

            >>> z.hessian([x, y])
            [[-0.08838835,  1.33381153],
             [ 1.33381153,  0.48045301]]

            >>> z.hessian([y, 3, 0.4, x, -19])
            [[ 0.48045301,  0.        ,  0.        ,  1.33381153,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
             [ 1.33381153,  0.        ,  0.        , -0.08838835,  0.        ],
             [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ]]

        See Also
        --------
        gradient, d, d2, d2c
        """
        try:
            hess = []
            for v1 in variables:
                hess.append([self.d2c(v1,v2) for v2 in variables])
        except TypeError:
            hess = [[self.d2(variables)]]
        return hess
        
    def sqrt(self):
        """
        A convenience function equal to x**0.5. This is required for some 
        ``numpy`` functions like ``numpy.sqrt``, ``numpy.std``, etc.
        """
        return self**0.5
        
    def _get_variables(self, ad_funcs):
        # List of involved variables (ADV objects):
        variables = set()
        for expr in ad_funcs:
            variables |= set(expr._lc)
        return variables
    
    def __add__(self, val):
        ad_funcs = map(to_auto_diff, (self, val))

        x = ad_funcs[0].x
        y = ad_funcs[1].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = x + y
        
        ########################################
        variables = self._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):
        lc_wrt_args = [1., 1.]
        qc_wrt_args = [0., 0.]
        cp_wrt_args = 0.

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.
        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs, variables, lc_wrt_args,
                                    qc_wrt_args, cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    
    def __radd__(self, val):
        """
        This method shouldn't need any modification if __add__ has
        been defined
        """
        return self+val

    def __mul__(self, val):
        ad_funcs = map(to_auto_diff, (self, val))

        x = ad_funcs[0].x
        y = ad_funcs[1].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = x*y
        
        ########################################

        variables = self._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):
        lc_wrt_args = [y, x]
        qc_wrt_args = [0., 0.]
        cp_wrt_args = 1.

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.
        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs, variables, lc_wrt_args,
                                    qc_wrt_args, cp_wrt_args)
                                    
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    
    def __rmul__(self, val):
        """
        This method shouldn't need any modification if __mul__ has
        been defined
        """
        return self*val    
    
    def __div__(self, val):
        return self.__truediv__(val)
    
    def __truediv__(self, val):
        ad_funcs = map(to_auto_diff, (self, val))

        x = ad_funcs[0].x
        y = ad_funcs[1].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = x/y
        
        ########################################

        variables = self._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):
        lc_wrt_args = [1./y, -x/y**2]
        qc_wrt_args = [0., 2*x/y**3]
        cp_wrt_args = -1./y**2

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.
        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs, variables, lc_wrt_args,
                                    qc_wrt_args, cp_wrt_args)
                                    
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    
    
    def __rdiv__(self, val):
        """
        This method shouldn't need any modification if __pow__ and __mul__ have
        been defined
        """
        return val*self**(-1)
    
    def __rtruediv__(self, val):
        """
        This method shouldn't need any modification if __pow__ and __mul__ have
        been defined
        """
        return val*self**(-1)
    
    def __sub__(self, val):
        """
        This method shouldn't need any modification if __add__ and __mul__ have
        been defined
        """
        return self + (-1*val)

    def __rsub__(self, val):
        """
        This method shouldn't need any modification if __add__ and __mul__ have
        been defined
        """
        return -1*self + val

    def __pow__(self, val):
        ad_funcs = map(to_auto_diff, (self, val))
        
        x = ad_funcs[0].x
        y = ad_funcs[1].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = x**y
        
        ########################################

        variables = self._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        if isinstance(x, complex) or isinstance(y, complex):
            if abs(x)>0 and ad_funcs[1].d(ad_funcs[1])!=0:
                lc_wrt_args = [y*x**(y-1), x**y*cmath.log(x)]
                qc_wrt_args = [y*(y-1)*x**(y-2), (cmath.log(x))**2]
                cp_wrt_args = x**(y-1)*(x*cmath.log(x)+y)
            else:
                lc_wrt_args = [y*x**(y-1), 0.]
                qc_wrt_args = [y*(y-1)*x**(y-2), 0.]
                cp_wrt_args = 0.
        else:
            if x>0 and ad_funcs[1].d(ad_funcs[1])!=0:
                lc_wrt_args = [y*x**(y-1), x**y*math.log(x)]
                qc_wrt_args = [y*(y-1)*x**(y-2), (math.log(x))**2]
                cp_wrt_args = x**(y-1)*(x*math.log(x)+y)
            else:
                lc_wrt_args = [y*x**(y-1), 0.]
                qc_wrt_args = [y*(y-1)*x**(y-2), 0.]
                cp_wrt_args = 0.
            

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.
        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs, variables, lc_wrt_args,
                                    qc_wrt_args, cp_wrt_args)
                                    
                                   
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)

    def __rpow__(self,val):
        return to_auto_diff(val)**self
        
    def __mod__(self, val):
        return self - val*_floor(self/val)
        
    def __rmod__(self, val):
        return val - self*_floor(val/self)
        
    def __neg__(self):
        return -1*self
    
    def __pos__(self):
        return self
        
    def __invert__(self):
        return -(self+1)

    def __abs__(self):
        ad_funcs = map(to_auto_diff, [self])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = abs(x)
        
        ########################################

        variables = self._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        # catch the x=0 exception
        try:
            lc_wrt_args = [x/abs(x)]
            qc_wrt_args = [1/abs(x)-(x**2)/abs(x)**3]
        except ZeroDivisionError:
            lc_wrt_args = [0.0]
            qc_wrt_args = [0.0]
            
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.
        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs, variables, lc_wrt_args,
                                    qc_wrt_args, cp_wrt_args)
                                    
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
        
    def toInt(self):
        """
        Converts the base number to an ``int`` object
        """
        self.x = int(self.x)
        return self

    def toFloat(self):
        """
        Converts the base number to an ``float`` object
        """
        self.x = float(self.x)
        return self
    
    def toLong(self):
        """
        Converts the base number to an ``long`` object
        """
        self.x = long(self.x)
        return self

    def toComplex(self):
        """
        Converts the base number to an ``complex`` object
        """
        self.x = complex(self.x)
        return self
    
    # coercion follows the capabilities of the respective input types
    def __int__(self):
        return int(self.x)
    
    def __float__(self):
        return float(self.x)
    
    def __long__(self):
        return long(self.x)
    
    def __complex__(self):
        return complex(self.x)
        
    # let the respective numeric types take care of the comparison operators
    def __eq__(self, val):
        ad_funcs = map(to_auto_diff, [self, val])
        return ad_funcs[0].x==ad_funcs[1].x
    
    def __ne__(self, val):
        return not self==val

    def __lt__(self, val):
        ad_funcs = map(to_auto_diff, [self, val])
        return ad_funcs[0].x<ad_funcs[1].x
    
    def __le__(self, val):
        return (self<val) or (self==val)
    
    def __gt__(self, val):
        ad_funcs = map(to_auto_diff, [self, val])
        return ad_funcs[0].x>ad_funcs[1].x
    
    def __ge__(self, val):
        return (self>val) or (self==val)
    
    def __nonzero__(self):
        return type(self.x).__nonzero__(self.x)
        
class ADV(ADF):
    """
    A convenience class for distinguishing between FUNCTIONS (ADF) and VARIABLES
    """
    def __init__(self, value, tag=None):
        # The first derivative of a variable wrt itself is always 1.0 and 
        # the second is always 0.0
        super(ADV, self).__init__(value, {self:1.0}, {self:0.0}, {}, tag=tag)

        # by generating this random trace, it should preserve relations even
        # after pickling and un-pickling (not sure this is working yet...)
        self._trace = randint(1,100000000)
        
def adnumber(x, tag=None):
    """
    Constructor of automatic differentiation (AD) variables, or numbers that
    keep track of the derivatives of subsequent calculations.
    
    Parameters
    ----------
    x : scalar or array-like
        The nominal value(s) of the variable(s). Any numeric type or array is
        supported. If ``x`` is another AD object, a fresh copy is returned that
        contains all the derivatives of ``x``, but is not related to ``x`` in 
        any way.
    
    Optional
    --------
    tag : str
        A string identifier. If an array of values for ``x`` is input, the tag 
        applies to all the new AD objects.
        
    Returns
    -------
    x_ad : an AD object
        
    Examples
    --------
    
    Creating an AD object (any numeric type can be input--int, float, complex,
    etc.)::
        
        >>> from ad import adnumber
        >>> x = adnumber(2)
        >>> x
        ad(2.0)
        >>> x.d(x)  # the derivative wrt itself is always 1.0
        1.0
    
        >>> y = adnumber(0.5, 'y')  # tags are nice for tracking AD variables
        >>> y
        ad(0.5, y)

    Let's do some math::
        
        >>> x*y
        ad(1.0)
        >>> x/y
        ad(4.0)
        
        >>> z = x**y
        >>> z
        ad(1.41421356237)
        
        >>> z.d(x)
        0.3535533905932738
        >>> z.d2(x)
        -0.08838834764831845
        >>> z.d2c(x, y)  # z.d2c(y, x) returns the same
        1.333811534061821
        >>> z.d2c(y, y)  # equivalent to z.d2(y)
        0.4804530139182014
        
        # only derivatives wrt original variables are tracked, thus the 
        # derivative of z wrt itself is zero
        >>> z.d(z)
        0.0
        
    We can also use the exponential, logarithm, and trigonometric functions::
        
        >>> from ad.admath import *  # sin, exp, etc. math funcs
        >>> z = sqrt(x)*sin(erf(y)/3)
        >>> z
        ad(0.24413683610889056)
        >>> z.d()
        {ad(0.5, y): 0.4080425982773223, ad(2.0): 0.06103420902722264}
        >>> z.d2()
        {ad(0.5, y): -0.42899113441354375, ad(2.0): -0.01525855225680566}
        >>> z.d2c()
        {(ad(0.5, y), ad(2.0)): 0.10201064956933058}

    We can also initialize multiple AD objects in the same constructor by
    supplying a sequence of values--the ``tag`` keyword is applied to all the
    new objects::
        
        >>> x, y, z = adnumber([2, 0.5, (1+3j)], tag='group1')
        >>> z
        ad((1+3j), group1)
    
    If ``numpy`` is installed, the returned array can be converted to a 
    ``numpy.ndarray`` using the ``numpy.array(...)`` constructor::
    
        >>> import numpy as np
        >>> x = np.array(adnumber([2, 0.5, (1+3j)])
        
    From here, many ``numpy`` operations can be performed (i.e., sum, max,
    etc.), though I haven't performed extensive testing to know which functions
    won't work.
        
    """
    try:
        # If the input is a numpy array, return a numpy array, otherwise try to
        # match the input type (numpy arrays are constructed differently using
        # numpy.array(...) and the actual class type, numpy.ndarray(...), so we
        # needed an exception). Other iterable types may need exceptions, but
        # this should always work for list and tuple objects at least.
        
        if numpy_installed and isinstance(x, numpy.ndarray):
            return numpy.array([adnumber(xi, tag) for xi in x])
        elif isinstance(x, (tuple, list)):
            return type(x)([adnumber(xi, tag) for xi in x])
        else:
            raise TypeError
        
    except TypeError:
        if isinstance(x, ADF):
            cp = copy.deepcopy(x)
            return cp
        elif isinstance(x, CONSTANT_TYPES):
            return ADV(x, tag)

    raise NotImplementedError(
        'Automatic differentiation not yet supported for {:} objects'.format(
        type(x))
        )

adfloat = adnumber  # for backwards compatibility

def gh(func):
    """
    Generates gradient (g) and hessian (h) functions of the input function 
    using automatic differentiation. This is primarily for use in conjunction
    with the scipy.optimize package, though certainly not restricted there.
    
    NOTE: If NumPy is installed, the returned object from ``grad`` and ``hess`` 
    will be a NumPy array. Otherwise, a generic list (or nested list, for 
    ``hess``) will be returned.
    
    Parameters
    ----------
    func : function
        This function should be composed of pure python mathematics (i.e., it
        shouldn't be used for calling an external executable since AD doesn't 
        work for that).
    
    Returns
    -------
    grad : function
        The AD-compatible gradient function of ``func``
    hess : function
        The AD-compatible hessian function of ``func``
        
    Examples
    --------
    ::
    
        >>> def my_cool_function(x):
        ...     return (x[0]-10.0)**2 + (x[1]+5.0)**2
        ...
        >>> grad, hess = gh(my_cool_function)
        >>> x = [24, 17]
        >>> grad(x)
        [28.0, 44.0]
        >>> hess(x)
        [[2.0, 0.0], [0.0, 2.0]]
        
        >>> import numpy as np
        >>> x_arr = np.array(x)
        >>> grad(x_arr)
        array([ 28.,  44.])
        >>> hess(x_arr)
        array([[ 2.,  0.],
               [ 0.,  2.]])
 
    """
    def grad(x, *args):
        xa = adnumber(x)
        if numpy_installed and isinstance(x, numpy.ndarray):
            ans = func(xa, *args)
            if isinstance(ans, numpy.ndarray):
                return numpy.array(ans[0].gradient(list(xa)))
            else:
                return numpy.array(ans.gradient(list(xa)))
        else:
            try:
                # first see if the input is an array-like object (list or tuple)
                return func(xa, *args).gradient(xa)
            except TypeError:
                # if it's a scalar, then update to a list for the gradient call
                return func(xa, *args).gradient([xa])
    
    def hess(x, *args):
        xa = adnumber(x)
        if numpy_installed and isinstance(x, numpy.ndarray):
            ans = func(xa, *args)
            if isinstance(ans, numpy.ndarray):
                return numpy.array(ans[0].hessian(list(xa)))
            else:
                return numpy.array(ans.hessian(list(xa)))
        else:
            try:
                # first see if the input is an array-like object (list or tuple)
                return func(xa, *args).hessian(xa)
            except TypeError:
                # if it's a scalar, then update to a list for the hessian call
                return func(xa, *args).hessian([xa])

    # customize the documentation with the input function name
    for f, name in zip([grad, hess], ['gradient', 'hessian']):
        f.__doc__ =  'The %s of %s, '%(name, func.__name__)
        f.__doc__ += 'calculated using automatic\ndifferentiation.\n\n'
        if func.__doc__ is not None and isinstance(func.__doc__, str):
            f.__doc__ += 'Original documentation:\n'+func.__doc__

    return grad, hess
        
