from setuptools import setup, find_packages

README_FILE = 'README.md'

project_name = "cola"
setup(
    name=project_name,
    description="",
    version="0.0.1",
    author="Marc Finzi and Andres Potapczynski",
    author_email="maf820@nyu.edu",
    license='MIT',
    python_requires='>=3.10',
    install_requires=[
        'pytest', 'tqdm>=4.38', 'matplotlib',
        # 'plum-dispatch @ git+ssh://git@github.com/beartype/plum.git',
        # 'plum-dispatch @ git+https://github.com/beartype/plum',
        'plum-dispatch @ git+https://github.com/mfinzi/plum'\
                '#537534597f0061ea38e499d5127e2fe78463cdfb',
    ],
    packages=find_packages(),
    # long_description=open('../README.md').read(),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/wilson-labs/cola',
    classifiers=[
        'Development Status :: 4 - Beta',
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
    ],
)
