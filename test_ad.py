# -*- coding: utf-8 -*-
"""
Created on Wed May 22 06:47:38 2013

@author: tisimst
"""
import math
import ad
from ad import __author__

def test_ad():
    print 'Running test suite on automatically differentiated objects:'
    print '> constructors.....',test_constructors()
    print '> comparitors......',test_comparitors()
    print '> math operators...',test_math_operators()
    print 'All tests passed successfully!'
    pass

def test_constructors():
    x = ad.AD(0)
    y = ad.AD(1,'y')
    
    assert x.tag   == None
    assert x.x     == 0.
    assert x.d(x)  == 1.
    assert x.d2(x) == 0.
    # the derivative functions should be able to handle any inputs
    for v in [1,0.5,dict(),list(),set()]:
        assert x.d(v)  == 0.
        assert x.d2(v) == 0.
    # a 'None' input returns the full first derivative dictionary
    assert {x:1.0}==x.d(None)
    
    assert y.tag   == 'y'
    
    return 'pass'

def test_comparitors():
    x = ad.AD(0)
    y = ad.AD(3.14)
    z = ad.AD(3)
    
    assert x<y
    assert x<=y
    assert x<3.14
    assert not x>y
    assert not x>=y
    assert x!=y
    assert x!=3.14
    assert y!=z
    assert y>z
    assert y>=z
    assert y>3
    assert x==x
    assert not z!=z
    assert z==3 # maintains loose relations to other numeric types
    
    return 'pass'

def test_math_operators():
    x = ad.AD(2)
    y = ad.AD(0.5)
    z = x+y
    w = z**2
    u = ad.sin(x)
    v = ad.exp(x)
    
    assert x == 4*y
    assert z == x+y
    assert w == (x+y)**2
    
    assert z.d(x) == 1.
    assert z.d(y) == 1.
    assert z.d2(x) == 0.
    assert z.d2(y) == 0.
    assert z.d2c(x,y) == 0.
    assert z.d2c(y,x) == 0.
    assert z.d2c(x,x) == 0. # special case handled as z.d2(x)
    
    assert w.d(z) == 0. # derivatives of ADF objects are not stored, only ADV
    assert w.d2(z) == 0.
    assert w.d(x) == 5.
    assert w.d(y) == 5.
    assert w.d2(x) == 2.
    assert w.d2(y) == 2.
    assert w.d2c(x,y) == 2.
    assert w.d2c(x,x) == w.d2(x)
    
    assert u.d(x) == math.cos(x.x)
    assert u.d2(x) == -math.sin(x.x)
    assert v.d(x) == v
    assert v.d2(x) == v
    assert ad.log(x) == ad.ln(x)
    assert ad.log10(x) == ad.log(x)/ad.ln(10) # log() and ln() are interchangeable
    
    return 'pass'
    
if __name__=='__main__':
    test_ad()