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
from unittest import TestCase, TestSuite

################################################################################

    
class AdTest:
    def setUp(self):
        self.x = adnumber(self.xi, tag='x')
        self.y = adnumber(self.yi)
    
    def test_tags(self):
        'test tag property'
        self.assertEqual(self.x.tag, 'x')
        self.assertTrue(self.y.tag is None)
    
    def test_comparisons(self):
        'test object comparisons'
        x,y = self.x, self.y
        
        self.assertEqual(x, 2)
        self.assertNotEqual(x, 1)
        self.assertTrue(x)  # nonzero
        self.assertTrue(x < 3)
        self.assertTrue(x <= 2)
        self.assertTrue(x > 1)
        self.assertTrue(x >= 2)
        
        self.assertEqual(y, 3)
        self.assertNotEqual(y, 2)
        self.assertTrue(y)  # nonzero
        self.assertTrue(y < 4)
        self.assertTrue(y <= 3)
        self.assertTrue(y > 2)
        self.assertTrue(y >= 3)
        
        # test underlying object comparisons
        self.assertEqual(x.x, 2)
        self.assertEqual(y.x, 3)
    
    def test_ADV_derivs(self):
        "test derivatives of ADV (independent variable) objects"
        x,y = self.x, self.y
        self.assertEqual(x.d(x), 1)
        self.assertEqual(y.d(y), 1)
        self.assertEqual(y.d(x), 0)
        self.assertEqual(x.d(y), 0)
        self.assertEqual(x.d(1), 0)
        self.assertEqual(y.d(1), 0)
        self.assertEqual(x.d2(x), 0)
        self.assertEqual(y.d2(y), 0)
        self.assertEqual(x.d2(y), 0)
        self.assertEqual(y.d2(x), 0)

    def test_ADF_derivs(self):
        'test derivatives of ADF (dependent variable) objects'
        x, y = self.x, self.y
        xi, yi = self.xi, self.yi
        z_add = x + y
        self.assertEqual(z_add, xi + yi)
        self.assertEqual(z_add.d(x), 1)
        self.assertEqual(z_add.d(y), 1)
        
        # dependent variables not traced
        self.assertEqual(z_add.d(z_add), 0)
        self.assertEqual(z_add.d2(x), 0)
        self.assertEqual(z_add.d2(y), 0)
        self.assertEqual(z_add.d2c(x, y), 0)
        self.assertEqual(z_add.d2c(y, x), z_add.d2c(x, y))
        self.assertEqual(z_add.d2c(x, z_add), 0)
        self.assertEqual(z_add.gradient([x, 1, y]), [1, 0, 1])

        z_sub = x - y
        self.assertEqual(z_sub, xi - yi)
        self.assertEqual(z_sub.d(x), 1)
        self.assertEqual(z_sub.d(y), -1)
        self.assertEqual(z_sub.d2(x), 0)
        self.assertEqual(z_sub.d2(y), 0)
        self.assertEqual(z_sub.d2c(x, y), 0)
        self.assertEqual(z_sub.gradient([x, y, z_add]), [1, -1, 0])

        z_mul = x*y
        self.assertEqual(z_mul, xi*yi)
        self.assertEqual(z_mul.d(x), 3)
        self.assertEqual(z_mul.d(y), 2)
        self.assertEqual(z_mul.d2(x), 0)
        self.assertEqual(z_mul.d2(y), 0)
        self.assertEqual(z_mul.d2c(x, y), 1)
        
        z_div = x/y
        self.assertEqual(z_div, xi/yi)
        self.assertEqual(z_div.d(x), 1./yi)
        self.assertEqual(z_div.d(y), -xi/(yi**2))
        self.assertEqual(z_div.d2(x), 0)
        self.assertEqual(z_div.d2(y), 2*xi/(yi**3))
        self.assertEqual(z_div.d2c(x, y), -1./9)
        
        z_pow = x**y
        self.assertEqual(z_pow, xi**yi)
        self.assertEqual(z_pow.d(x), 12)
        self.assertEqual(z_pow.d(y), (8*math.log(2)))
        self.assertEqual(z_pow.d2(x), 12)
        self.assertEqual(z_pow.d2(y), (8*math.log(2)**2))
        self.assertEqual(z_pow.d2c(x, y), (4 + 12*math.log(2)))
        self.assertEqual(z_pow.hessian([z_mul, y, x]), [
            [0,                  0,                  0], 
            [0,   8*math.log(2)**2, 4 + 12*math.log(2)], 
            [0, 4 + 12*math.log(2),                 12]])
        
        z_mod = x%y
        self.assertEqual(z_mod, (x - y*ad._floor(x/y)))
        
        z_neg = -x
        self.assertEqual(z_neg, -1*x.x)
        
        z_pos = +x
        self.assertEqual(z_pos, x.x)
        
        z_inv = ~x
        self.assertEqual(z_inv, -(x+1))
        
        z_abs = abs(-x.x)
        self.assertEqual(z_abs, x)
    
    def test_coercion(self):
        'test coercion methods'
        x = self.x
        if isinstance(x.x, (int, float)):
            msg = '{:} and {:}'.format(int(x), type(int(x)))
            self.assertEqual(int(x), 2, msg)
            self.assertTrue(isinstance(int(x), int), msg)
             
            msg = '{:} and {:}'.format(float(x), type(float(x)))
            self.assertEqual(float(x), 2.0)
            self.assertTrue(isinstance(float(x), float))
            
            msg = '{:} and {:}'.format(complex(x), type(complex(x)))    
            self.assertEqual(complex(x), 2+0j)
            self.assertTrue(isinstance(complex(x), complex))


    def test_trace(self):
        'test trace_me'
        z_add = self.x + self.y
        z_add.trace_me()
        self.assertEqual(z_add.d(z_add), 1)
        self.assertEqual(z_add.d2(z_add), 0)
    
    def test_gh(self):
        'test gh function wrapper'
        x, y = self.x, self.y
        def test_func(x, a):
            return (x[0] + x[1])**a
        testg, testh = gh(test_func)
        self.assertEqual(testg([x, y], 3), ((x + y)**3).gradient([x, y]))
        self.assertEqual(testh([x, y], 3), ((x + y)**3).hessian([x, y]))

    def test_jacobian(self):
        'test jacobian function'
        x, y = self.x, self.y
        self.assertEqual(jacobian([x*y, x+y], [x, 1, y]),
                    [[3.0, 0.0, 2.0], [1.0, 0.0, 1.0]])
    

class AdTestInt(AdTest, TestCase):
    xi, yi = (2, 3)

class AdTestFloat(AdTest, TestCase):
    xi, yi = (2.0, 3.0)

if __name__ == '__main__':
    unittest.main()
