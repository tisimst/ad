import os, sys
#from setuptools import setup
import distutils.core

# Building through 2to3, for Python 3 (see also setup(...,
# cmdclass=...), below:
try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    # 2.x
    from distutils.command.build_py import build_py
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

extras = {}
if sys.version_info >= (3,):
    extras['use_2to3'] = True

readme = 'README'

distutils.core.setup(
    name='ad',
    version='1.2.1',
    author='Abraham Lee',
    author_email='tisimst@gmail.com',
    description='Fast, transparent first- and second-order automatic differentiation',
    url='http://pythonhosted.org/ad',
    license='BSD License',
    long_description=read(readme),
    package_data={'': [readme]},
    #include_package_data=True,
    packages=['ad', 'ad.admath', 'ad.linalg'],
    keywords=[
        'automatic differentiation',
        'first order',
        'second order',
        'derivative',
        'algorithmic differentiation',
        'computational differentiation',
        'optimization',
        'linear algebra'
        ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
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
        ],
    cmdclass={'build_py': build_py},
    **extras
    )
