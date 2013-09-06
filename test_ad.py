"""
================================================================================
ad test suite
================================================================================

Author: Abraham Lee
Copyright: 2013

"""
import ad
from ad import *
from ad.admath import *
import math
import cmath

################################################################################
for xi, yi in zip((2, 2.0), (3, 3.0)):
    
    # variables to test
    x = adnumber(xi, tag='x')
    y = adnumber(yi)
    
    # test tag property
    assert x.tag=='x'
    assert y.tag==None
    
    # test object comparisons
    assert x==2
    assert x!=1
    assert x  # nonzero
    assert x<3
    assert x<=2
    assert x>1
    assert x>=2
    
    assert y==3
    assert y!=2
    assert y  # nonzero
    assert y<4
    assert y<=3
    assert y>2
    assert y>=3
    
    # test underlying object comparisons
    assert x.x==2
    assert y.x==3
    
    # test derivatives of ADV (independent variable) objects
    assert x.d(x)==1
    assert y.d(y)==1
    assert y.d(x)==0
    assert x.d(y)==0
    assert x.d(1)==0
    assert y.d(1)==0
    assert x.d2(x)==0
    assert y.d2(y)==0
    assert x.d2(y)==0
    assert y.d2(x)==0
    
    # test derivatives of ADF (dependent variable) objects
    z_add = x + y
    assert z_add==xi + yi, z_add
    assert z_add.d(x)==1, z_add.d(x)
    assert z_add.d(y)==1, z_add.d(y)
    assert z_add.d(z_add)==0, z_add.d(z_add) # dependent variables not traced
    assert z_add.d2(x)==0, z_add.d2(x)
    assert z_add.d2(y)==0, z_add.d2(y)
    assert z_add.d2c(x, y)==0, z_add.d2c(x, y)
    assert z_add.d2c(y, x)==z_add.d2c(x, y), z_add.d2c(y, x)
    assert z_add.d2c(x, z_add)==0, z_add.d2c(x, z_add)
    assert z_add.gradient([x, 1, y])==[1, 0, 1], z_add.gradient([x, 1, y])

    z_sub = x - y
    assert z_sub==xi - yi, z_sub
    assert z_sub.d(x)==1, z_sub.d(x)
    assert z_sub.d(y)==-1, z_sub.d(y)
    assert z_sub.d2(x)==0, z_sub.d2(x)
    assert z_sub.d2(y)==0, z_sub.d2(y)
    assert z_sub.d2c(x, y)==0, z_sub.d2c(x, y)
    assert z_sub.gradient([x, y, z_add])==[1, -1, 0], z_sub.gradient([x, y, z_add])

    z_mul = x*y
    assert z_mul==xi*yi, z_mul
    assert z_mul.d(x)==3, z_mul.d(x)
    assert z_mul.d(y)==2, z_mul.d(y)
    assert z_mul.d2(x)==0, z_mul.d2(x)
    assert z_mul.d2(y)==0, z_mul.d2(y)
    assert z_mul.d2c(x, y)==1, z_mul.d2c(x, y)
    
    z_div = x/y
    assert z_div==xi/yi, z_div
    assert z_div.d(x)==1./yi, z_div.d(x)
    assert z_div.d(y)==-xi/(yi**2), z_div.d(y)
    assert z_div.d2(x)==0, z_div.d2(x)
    assert z_div.d2(y)==2*xi/(yi**3), z_div.d2(y)
    assert z_div.d2c(x, y)==-1./9, z_div.d2c(x, y)
    
    z_pow = x**y
    assert z_pow==xi**yi, z_pow
    assert z_pow.d(x)==12, z_pow.d(x)
    assert z_pow.d(y)==(8*math.log(2)), z_pow.d(y)
    assert z_pow.d2(x)==12, z_pow.d2(x)
    assert z_pow.d2(y)==(8*math.log(2)**2), z_pow.d2(y)
    assert z_pow.d2c(x, y)==(4 + 12*math.log(2)), z_pow.d2c(x, y)
    assert z_pow.hessian([z_mul, y, x])==[
        [0,                  0,                  0], 
        [0,   8*math.log(2)**2, 4 + 12*math.log(2)], 
        [0, 4 + 12*math.log(2),                 12]], \
        z_pow.hessian([z_mul, y, x])
    
    z_mod = x%y
    assert z_mod==(x - y*ad._floor(x/y)), z_mod
    
    z_neg = -x
    assert z_neg==-1*x.x, z_neg
    
    z_pos = +x
    assert z_pos==x.x, z_pos
    
    z_inv = ~x
    assert z_inv==-(x+1), z_inv
    
    z_abs = abs(-x.x)
    assert z_abs==x, z_abs
    
    # test coercion methods
    if isinstance(x.x, (int, float)):
        assert int(x)==2 and isinstance(int(x), int), '{:} and {:}'.format(
            int(x), type(int(x)))
        assert float(x)==2.0 and isinstance(float(x), float), \
            '{:} and {:}'.format(float(x), type(float(x)))
    assert complex(x)==2+0j and isinstance(complex(x), complex), \
        '{:} and {:}'.format(complex(x), type(complex(x)))
    
    # test trace_me
    z_add.trace_me()
    z_add.d(z_add)==1, z_add.d(z_add)
    z_add.d2(z_add)==0, z_add.d2(z_add)
    
    # test gh function wrapper
    def test_func(x, a):
        return (x[0] + x[1])**a
    testg, testh = gh(test_func)
    assert testg([x, y], 3)==((x + y)**3).gradient([x, y]), testg([x, y], 3)
    assert testh([x, y], 3)==((x + y)**3).hessian([x, y]), testh([x, y], 3)

    # test jacobian function
    assert jacobian([z_mul, z_add], [x, 1, y])==[[3.0, 0.0, 2.0], 
                                                 [1.0, 0.0, 1.0]], \
        jacobian([z_mul, z_add], [x, 1, y])
    
    print('** All tests passed successfully for xi={:} and yi={:}!'.format(xi, yi))