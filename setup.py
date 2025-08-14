"""Setup configuration for LLSF package."""

from setuptools import setup, find_packages
import os

# Read version from version.py
def get_version():
    """Get version from version.py file."""
    version_file = os.path.join(os.path.dirname(__file__), 'llsf', 'version.py')
    version_vars = {}
    with open(version_file) as f:
        exec(f.read(), version_vars)
    return version_vars['__version__']

# Read README
def get_long_description():
    """Get long description from README.md."""
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    try:
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "LLSF: Learning Label-Specific Features for Multi-Label Classification"

setup(
    name='llsf',
    version=get_version(),
    description='Learning Label-Specific Features (LLSF) for Multi-Label Classification',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Prady029',
    author_email='prady029@duck.com',
    url='https://github.com/Prady029/LLSF-Learning-Label-Specific-Features-for-Multi-Label-Classifcation',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'scikit-learn>=0.24.0',
        'scipy>=1.5.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=21.0',
            'flake8>=3.8',
            'mypy>=0.800',
        ],
        'docs': [
            'sphinx>=3.0',
            'sphinx-rtd-theme>=0.5',
        ],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='machine-learning multi-label-classification scikit-learn llsf',
    project_urls={
        'Bug Reports': 'https://github.com/Prady029/LLSF-Learning-Label-Specific-Features-for-Multi-Label-Classifcation/issues',
        'Source': 'https://github.com/Prady029/LLSF-Learning-Label-Specific-Features-for-Multi-Label-Classifcation',
    },
)
