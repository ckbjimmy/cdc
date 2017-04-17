from setuptools import setup, find_packages

setup(name='cdc',
      version='0.1',
      description='Clinical Document Classification Pipeline',
      url='http://github.com/ckbjimmy/cdc',
      author='Wei-Hung Weng',
      author_email='ckbjimmy@hms.harvard.edu',
      license='MIT',
      packages=['cdc'],
      install_requires=["pandas", "xmltodict", "numpy", "scipy", "scikit-learn", "nltk", "beautifulsoup4", "gensim", "psutil"],
      zip_safe=False)
