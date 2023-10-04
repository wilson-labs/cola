import io
import os
import re

from setuptools import find_packages, setup


# Get version from setuptools_scm file
def find_version(*file_paths):
    try:
        with io.open(os.path.join(os.path.dirname(__file__), *file_paths), encoding="utf8") as fp:
            version_file = fp.read()
            pattern = r"^__version__ = version = ['\"]([^'\"]*)['\"]"
        version_match = re.search(pattern, version_file, re.M)
        return version_match.group(1)
    except Exception:
        return None


README_FILE = 'README.md'

project_name = "cola-ml"
setup(
    name=project_name,
    description="",
    version=find_version("cola", "version.py"),
    author="Marc Finzi and Andres Potapczynski",
    author_email="maf820@nyu.edu",
    license='MIT',
    python_requires='>=3.10',
    install_requires=[
        'scipy',
        'tqdm>=4.38',
        'cola-plum-dispatch==0.1.1',
        'optree',
        'pytreeclass',
    ],
    extras_require={
        'dev': ['pytest', 'pytest-cov', 'setuptools_scm', 'pre-commit'],
    },
    packages=find_packages(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wilson-labs/cola',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=[
        'linear algebra',
        'linear ops',
        'sparse',
        'PDE',
        'AI',
        'ML',
    ],
)
