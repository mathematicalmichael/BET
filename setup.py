#!/usr/bin/env python

# Copyright (C) 2014-2015 BET Development Team

'''
The python script for building the BET package and subpackages.
'''
try:
  from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='bet',
      version='2.0.0',
      description='Butler, Estep, Tavener method',
      author='Steven Mattis',
      author_email='steve.a.mattis@gmail.com',
      license='GNU LGPL',
      url='https://github.com/UT-CHG/BET',
      packages=['bet', 'bet.sampling', 'bet.calculateP', 'bet.postProcess', 'bet.sensitivity'],
      install_requires=['matplotlib<=2.2.3', 'pyDOE', 'scipy<=0.19.1',
          'numpy<1.13.0', 'nose'])
