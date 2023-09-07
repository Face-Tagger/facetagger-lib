from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="face_tagger",
    version="1.0.0",
    description="Python library designed to classify photos containing specific individuals from a collection of "
                "images.",
    author="Sohn YoungJin",
    author_email="sonkim1001@naver.com",
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
