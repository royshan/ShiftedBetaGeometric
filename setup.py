from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'scipy',
    'pandas',
    ]

setup(name='',
      version='0.1',
      description='',
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
          ],
      author='Fernando Nogueira',
      author_email='fmfnogueira@gmail.com',
      url='',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      )
