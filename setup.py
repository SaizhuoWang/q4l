from setuptools import find_packages, setup

setup(
    name="q4l",
    version="0.0.1",
    description="Quant 4.0 library",
    author="IDEA FinAI",
    license="MIT",
    include_package_data=True,
    packages=find_packages(
        ".",
        exclude=(
            "doc",
            "scripts",
            "tests",
            "examples",
        ),
    ),
    install_requires=[
        "pre-commit",
        "black==22.12.0",
        "isort==5.12.0",
        "docformatter==1.5.1",
        "autoflake==2.0.2",
        "blacken-docs==1.12.1",
        "flake8==6.0.0",
        "smart_open",
        "torch>=1.13.0",
        "submitit>=1.4.5",
        "lightning>2.0.3",
        "dgl>=1.0.1",
        "pandas>=1.5.0",
        "numpy>=1.23.5",
        "sphinx",
        "sphinx_rtd_theme",
        "hydra-core>=1.3",
        "yahooquery",
        "loky",
        "einops",
        "seaborn",
        "torchinfo",
        "pymongo>=4.5.0"
    ],
    dependency_links=[],
)
