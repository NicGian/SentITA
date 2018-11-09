from setuptools import setup, find_packages

setup(name='sentita',
      version='0.2.0',
      description='Sentiment polarity tool for Italian',
      url='http://github.com/nicgian/sentita',
      author='Giancarlo Nicola',
      author_email='giancarlo.nicola01@universitadipavia.it',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=[
          'keras', 'spacy', 'numpy',
      ],
      zip_safe=False)