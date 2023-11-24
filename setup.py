import setuptools

setuptools.setup(
    name="pywheat",
    version="0.0.9",
    license='GPLv3+',
    author="Ernesto Giron Echeverry",
    author_email="e.giron.e@gmail.com",
    description="Python library for simulation of wheat phenological development, crop growth and yield at large scales",
    keywords="wheat, crop modeling",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/egiron/pywheat",
    packages=setuptools.find_packages(include=["pywheat", "pywheat.*"]),
    python_requires='>=3.9',
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "pywheat=pywheat.main:cli",
        ],
    },
    install_requires=[
        'numpy',
        'numba',
        'pandas',
        'scikit-learn',
        'scipy',
        'tqdm',
        'seaborn',
        'Shapely',
        'ipython',
        'duckdb',
        'pyarrow',
        'click'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        "Programming Language :: Python :: 3",
        "License :: Free for non-commercial use",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
        "Environment :: Console",
        "Framework :: Jupyter",
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research"
    ],
    project_urls={
        'Issue Tracking': 'https://github.com/egiron/pywheat/issues',
    },
)