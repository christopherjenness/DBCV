#!/usr/bin/env python
from setuptools import setup, find_packages

DISTNAME = 'DBCV'
DESCRIPTION = 'Density Based Clustering Validation'
MAINTAINER = 'Christopher Jenness'
URL = 'https://github.com/christopherjenness/dbcv'

classifiers = ['Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.3']

with open('requirements.txt') as f:
    install_reqs = f.read().splitlines()

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          description=DESCRIPTION,
          packages=find_packages(),
          url=URL,
          classifiers=classifiers,
          install_requires=install_reqs)
