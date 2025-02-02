import os
from setuptools import setup, find_packages
import subprocess
import logging

setup(
    name='autosrec',
    version='0.0.1',
    description='AutoSRec: An Automated Sequential Recommender System',
    author='DATA Lab@Texas A&M University',
    author_email='suil0109@gmail.com',
    url='https://github.com/suil0109/AutoSRec.git',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    # package_data={
    #     'tods': ['resources/.entry_points.ini', 
    #             'resources/.requirements.txt',
    #              'resources/default_pipeline.json'
    #              ]
    # },
    install_requires=[
        'numpy==1.17.3',
        'pandas==0.25.2',
        'pytest==5.2.2',
        'PyYAML==5.1.2',
        'scikit-learn==0.21.3',
        'scipy==1.3.1',
        'tabulate==0.8.5',
        'tensorboard==2.0.0',
        'tensorflow-gpu==2.2.0',
        'termcolor==1.1.0',
        'terminaltables==3.1.0',
        'tqdm==4.36.1',
        'colorama==0.4.3',
    ],

)

