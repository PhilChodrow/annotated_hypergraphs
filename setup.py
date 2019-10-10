from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="ahyper",
    version="0.1",
    description="""Software for the analysis of Annotated Hypergraphs.""",
    long_description=readme(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
    keywords="hypergraphs ensemble shuffle network labelled higher-order",
    project_urls={
        "Source": "https://github.com/PhilChodrow/annotated_hypergraphs",
        "Tracker": "https://github.com/PhilChodrow/annotated_hypergraphs/issues",
    },
    author="Phil Chodrow & Andrew Mellor",
    author_email="mellor91@hotmail.co.uk",
    license="Apache Software License",
    packages=["ahyper"],
    python_requires=">=3",
    install_requires=["networkx", "pandas", "numpy", "matplotlib"],
    include_package_data=True,
    zip_safe=False,
    test_suite="nose.collector",
    tests_require=["nose"],
)
