from distutils.core import setup

with open('README.rst') as file:
    long_description = file.read()

setup(name='ad',
    version='1.1.1',
    author='Abraham Lee',
    description='Fast, transparent first- and second-order automatic differentiation',
    author_email='tisimst@gmail.com',
    url='https://github.com/tisimst/ad',
    license='BSD License',
    long_description=long_description,
    package_dir={'ad':''},
    packages=['ad'],
    include_package_data = True,
    package_data = {
        '': ['revision_history.txt', 'README.rst'],
        },
    keywords=[
        'automatic differentiation',
        'first order',
        'second order',
        'derivative'],
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
        ]
    )
