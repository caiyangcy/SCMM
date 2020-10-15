import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SC2MM", # Replace with your own username
    version="0.0.3",
    author="Cai Yang",
    author_email="u6625166@anu.edu.au",
    description="A package for StarCraft2 Micro-Management.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/caiyangcy/SC2DC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.10',
    install_requires=[
        'pysc2>=3.0.0',
        's2clientprotocol>=4.10.1.75800.0',
        'absl-py>=0.1.0',
        'numpy>=1.10',
    ],
)