import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SC2MM", 
    version="0.2.0",
    author="Cai Yang",
    author_email="u6625166@anu.edu.au",
    description="A package for StarCraft2 Micro-Management.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/caiyangcy/SCMM",
    packages=[
        'scmm',
        'scmm.env',
        'scmm.env.micro_env',
        'scmm.env.micro_env.maps',
        'scmm.bin',
        'scmm.agents',
        'scmm.agents.nn',
        'scmm.agents.genetic',
        'scmm.agents.potential_fields',
        'scmm.agents.scipted',
        'scmm.utils'
    ],
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
        'numpy==1.19.3', # 1.19.4 has a bug causing failure on sanity check
        'matplotlib==3.2.2',
        'pytorch==1.4.0'
    ],
)