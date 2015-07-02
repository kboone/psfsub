from setuptools import setup, find_packages
import os

setup(
    name='psfsub',
    version='0.0',
    description='Image subtractor optimized to find point sources',
    author='Kyle Boone',
    author_email='kboone@berkeley.edu',
    packages=find_packages(),
    scripts=['scripts/' + f for f in os.listdir('scripts')],
)
