from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='plt',
      version="0.0.1",
      description="PLT Model (api_pred)",
      author="Nous",
      #url="https://github.com/houssam0812/AI-Powered-Language-Testing",
      install_requires=requirements,
      packages=find_packages())