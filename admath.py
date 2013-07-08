# -*- coding: utf-8 -*-
"""
Mathematical operations that generalize many operations from the
standard math module so that they also track first and second derivatives.

Examples:

  from admath import sin
  
  # Manipulation of numbers with uncertainties:
  x = ad.AD(3)
  print sin(x)  # prints ADF(0.141120008...)

  # The umath functions also work on regular Python floats:
  print sin(3)  # prints 0.141120008...  This is a Python float.

Importing all the functions from this module into the global namespace
is possible.  This is encouraged when using a Python shell as a
calculator.  Example:

  import ad
  from ad.admath import *  # Imports tan(), etc.
  
  x = ad.AD(3, 0.1)
  print tan(x)  # tan() is the ad.admath.tan function

The numbers with derivative tracking handled by this module are objects from
the ad (automatic differentiation) module, from either the ADV or the ADF class.

(c) 2013 by Abraham D. Lee (ADL) <tisimst@gmail.com>.
Please send feature requests, bug reports, or feedback to this address.

This software is released under a dual license.  (1) The BSD license.
(2) Any other license, as long as it is obtained from the original
author.'''
"""
from __future__ import division
import math
from ad import __author__,ADF,to_auto_diff,_calculate_derivatives

__all__ = [
    'sin','asin','sinh','asinh',
    'cos','acos','cosh','acosh',
    'tan','atan','tanh','atanh',
    'exp','expm1',
    'erf','erfc',
    'factorial','gamma','lgamma',
    'log','ln','log10','log1p',
    'sqrt','hypot','pow',
    'degrees','radians',
    'ceil','floor','trunc','fabs','abs',
    # other miscellaneous functions that are conveniently defined
    'csc','acsc','csch','acsch',
    'sec','asec','sech','asech',
    'cot','acot','coth','acoth'
    ]

            
### FUNCTIONS IN THE MATH MODULE ##############################################
#
# Currently, there is no implementation for the following math module methods:
# - copysign
# - factorial <- depends on gamma
# - fmod
# - frexp
# - fsum
# - gamma* <- currently uses high-accuracy finite difference derivatives
# - lgamma <- depends on gamma
# - isinf
# - isnan
# - ldexp
# - modf
#
# we'll see if they(*) get implemented

# for convenience, (though totally defeating the purpose of having an 
# automatic differentiation class, define a fourth-order finite difference 
# derivative for when an analytical derivative doesn't exist
eps = 1e-8 # arbitrarily chosen
def _fourth_order_first_fd(func,x):
    fm2 = func(x-2*eps)
    fm1 = func(x-eps)
    fp1 = func(x+eps)
    fp2 = func(x+2*eps)
    return (fm2-8*fm1+8*fp1-fp2)/12/eps

def _fourth_order_second_fd(func,x):
    fm2 = func(x-2*eps)
    fm1 = func(x-eps)
    f   = func(x)
    fp1 = func(x+eps)
    fp2 = func(x+2*eps)
    return (-fm2+16*fm1-30*f+16*fp1-fp2)/12/eps**2
    
def abs(x):
    """
    The absolute value of a number, modified to work with AD objects
    """
    return fabs(x)
    
def acos(x):
    """
    The equivalent math.acos function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.acos(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [-1/sqrt(1-x**2)]
        qc_wrt_args = [(sqrt(1-x**2)*(-x))/(x**4-2*x**2+1)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [acos(xi) for xi in x]
        except TypeError:
            return math.acos(x)

def acosh(x):
    """
    The equivalent math.acosh function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.acosh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [-math.sin(x)]
        qc_wrt_args = [-math.cos(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [acosh(xi) for xi in x]
        except TypeError:
            return math.acosh(x)

def asin(x):
    """
    The equivalent math.asin function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.asin(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1/sqrt(1-x**2)]
        qc_wrt_args = [-(sqrt(1-x**2)*(-x))/(x**4-2*x**2+1)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [asin(xi) for xi in x]
        except TypeError:
            return math.asin(x)

def asinh(x):
    """
    The equivalent math.asinh function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.asinh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [math.cos(x)]
        qc_wrt_args = [-math.sin(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [asinh(xi) for xi in x]
        except TypeError:
            return math.asinh(x)

def atan(x):
    """
    The equivalent math.atan function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.atan(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1/(x**2+1)]
        qc_wrt_args = [-2*x/(x**4+x**2+1)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [atan(xi) for xi in x]
        except TypeError:
            return math.atan(x)

def atanh(x):
    """
    The equivalent math.atanh function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.atanh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1./(math.cos(x))**2]
        qc_wrt_args = [2*math.sin(x)/(math.cos(x))**3]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [atanh(xi) for xi in x]
        except TypeError:
            return math.atanh(x)

def ceil(x):
    """
    The equivalent math.ceil function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.ceil(x)
        
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

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [ceil(xi) for xi in x]
        except TypeError:
            return math.ceil(x)

def cos(x):
    """
    The equivalent math.cos function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.cos(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [-math.sin(x)]
        qc_wrt_args = [-math.cos(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [cos(xi) for xi in x]
        except TypeError:
            return math.cos(x)

def cosh(x):
    """
    The equivalent math.cosh function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.cosh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [math.sinh(x)]
        qc_wrt_args = [math.cosh(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [cosh(xi) for xi in x]
        except TypeError:
            return math.cosh(x)

def degrees(x):
    """
    The equivalent math.degrees function, modified to work with AD objects.
    """
    return (180/math.pi)*x

def erf(x):
    """
    The equivalent math.erf function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.erf(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [2*math.exp(-x**2)/math.sqrt(math.pi)]
        qc_wrt_args = [-4*x*math.exp(-x**2)/math.sqrt(math.pi)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [erf(xi) for xi in x]
        except TypeError:
            return math.erf(x)

def erfc(x):
    """
    The equivalent math.erfc function, modified to work with AD objects.
    """
    return 1-erf(x)
    
def exp(x):
    """
    The equivalent math.exp function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.exp(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [math.exp(x)]
        qc_wrt_args = [math.exp(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [exp(xi) for xi in x]
        except TypeError:
            return math.exp(x)

def expm1(x):
    """
    The equivalent math.expm1 function, modified to work with AD objects.
    """
    return exp(x)-1
    
def fabs(x):
    """
    The equivalent math.fabs function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.fabs(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        # catch the x=0 exception
        try:
            lc_wrt_args = [x/math.fabs(x)]
            qc_wrt_args = [1/math.fabs(x)-(x**2)/math.fabs(x)**3]
        except ZeroDivisionError:
            lc_wrt_args = [0.0]
            qc_wrt_args = [0.0]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [fabs(xi) for xi in x]
        except TypeError:
            return math.fabs(x)

def factorial(x):
    """
    The equivalent math.factorial function, modified to work with AD objects.
    """
    return gamma(x+1)

def floor(x):
    """
    The equivalent math.floor function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.floor(x)
        
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

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [floor(xi) for xi in x]
        except TypeError:
            return math.floor(x)

def gamma(x):
    """
    The equivalent math.gamma function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f   = math.gamma(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [_fourth_order_first_fd(math.gamma,x)]
        qc_wrt_args = [_fourth_order_second_fd(math.gamma,x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [gamma(xi) for xi in x]
        except TypeError:
            return math.gamma(x)

def hypot(x,y):
    """
    The equivalent math.hypot function, modified to work with AD objects.
    """
    return sqrt(x*x+y*y)
    
def lgamma(x,y):
    """
    The equivalent math.lgamma function, modified to work with AD objects.
    """
    return log(fabs(gamma(x)))
    
def log(x):
    """
    The equivalent math.log function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f   = math.log(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1/x]
        qc_wrt_args = [-1./x**2]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [log(xi) for xi in x]
        except TypeError:
            return math.log(x)

def log10(x):
    """
    The equivalent math.log10, modified to work with AD objects
    """
    return log(x)/log(10)

def log1p(x):
    """
    The equivalent math.log1p function, modified to work with AD objects.
    """
    return log(1+x)
    
def pow(x,y):
    """
    The equivalent math.pow function, modified to work with AD objects.
    """
    return x**y

def radians(x):
    """
    The equivalent math.radians function, modified to work with AD objects.
    """
    return (math.pi/180)*x
    
def sin(x):
    """
    The equivalent math.sin function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.sin(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [math.cos(x)]
        qc_wrt_args = [-math.sin(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [sin(xi) for xi in x]
        except TypeError:
            return math.sin(x)

def sinh(x):
    """
    The equivalent math.sinh function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.sinh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [math.cosh(x)]
        qc_wrt_args = [math.sinh(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [sinh(xi) for xi in x]
        except TypeError:
            return math.sinh(x)

def sqrt(x):
    """
    A equivalent math.sqrt function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.sqrt(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1./(2*math.sqrt(x))]
        qc_wrt_args = [-1./(4*(math.sqrt(x))**3)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [sqrt(xi) for xi in x]
        except TypeError:
            return math.sqrt(x)
            
def tan(x):
    """
    The equivalent math.tan function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.tan(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1./(math.cos(x))**2]
        qc_wrt_args = [2*math.sin(x)/(math.cos(x))**3]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [tan(xi) for xi in x]
        except TypeError:
            return math.tan(x)

def tanh(x):
    """
    The equivalent math.tanh function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.tanh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1./(math.cosh(x))**2]
        qc_wrt_args = [-2*math.sinh(x)/(math.cosh(x))**3]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [tanh(xi) for xi in x]
        except TypeError:
            return math.tanh(x)

def trunc(x):
    """
    The equivalent math.trunc function, modified to work with AD objects.
    """
    if isinstance(x,ADF):
        ad_funcs = map(to_auto_diff,[x])

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = math.trunc(x)
        
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

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _calculate_derivatives(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [trunc(xi) for xi in x]
        except TypeError:
            return math.trunc(x)

### OTHER CONVENIENCE FUNCTIONS ###############################################

def csc(x):
    """
    The cosecant function, modified to work with AD objects.
    """
    return 1/sin(x)
    
def sec(x):
    """
    The secant function, modified to work with AD objects.
    """
    return 1/cos(x)
    
def cot(x):
    """
    The cotangent function, modified to work with AD objects.
    """
    return 1/tan(x)

def csch(x):
    """
    The hyperbolic cosecant function, modified to work with AD objects.
    """
    return 1/sinh(x)
    
def sech(x):
    """
    The hyperbolic secant function, modified to work with AD objects.
    """
    return 1/cosh(x)
    
def coth(x):
    """
    The hyperbolic cotangent function, modified to work with AD objects.
    """
    return 1/tanh(x)

def acsc(x):
    """
    The arccosecant function, modified to work with AD objects.
    """
    return asin(1/x)

def asec(x):
    """
    The arcsecant function, modified to work with AD objects.
    """
    return acos(1/x)
    
def acot(x):
    """
    The arccotangent function, modified to work with AD objects.
    """
    return atan(1/x)

def acsch(x):
    """
    The hyperbolic arccosecant function, modified to work with AD objects.
    """
    return asinh(1/x)

def asech(x):
    """
    The hyperbolic arcsecant function, modified to work with AD objects.
    """
    return acosh(1/x)
    
def acoth(x):
    """
    The hyperbolic arccotangent function, modified to work with AD objects.
    """
    return atanh(1/x)

def ln(x):
    """
    A convenience natural logarithm function equal to math.log, modified to
    work with AD objects.
    """
    return log(x)

def logB(x,base):
    """
    A convenience logarithm function for any base number system, modified to
    work with AD objects.
    
    Note: base must be an integer >= 2
    """
    assert isinstance(base,int), 'base must be an integer'
    assert base>1, 'base must be at least 2'
    
    return log(x)/log(base)
    
