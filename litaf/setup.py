from setuptools import setup

setup(name='litaf',
      version='1.0',
      description='Package for customization of input data for AlphaFold and AlphaFold-Multimer',
      url='https://github.com/LIT-CCM-lab/LIT-AlphaFold',
      author='Luca Chiesa',
      author_email='luca.chiesa@unistra.com',
      license='',
      packages=['litaf'],
      install_requires=['colabfold==1.5.3'],
      zip_safe=False)