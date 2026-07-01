from setuptools import setup, find_packages

setup(
    name="ds_project_1",
    version="1.0.0",
    description="Advanced EDA & Feature Engineering Pipeline — DecodeLabs Batch 2026",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "openpyxl>=3.1",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "run-pipeline=main:run_pipeline",
        ]
    },
)
