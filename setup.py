import os
from setuptools import setup
import ad

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='ad',
    version=ad.__version__,
    author='Abraham Lee',
    author_email='tisimst@gmail.com',
    description='Fast, transparent first- and second-order automatic differentiation',
    url='http://pythonhosted.org/ad',
    license='BSD License',
    long_description=read('README.rst'),
    packages=['ad', 'ad.admath'],
    keywords=[
        'automatic differentiation',
        'first order',
        'second order',
        'derivative',
        'algorithmic differentiation',
        'computational differentiation',
        'optimization'
        ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities'
        ]
    )
