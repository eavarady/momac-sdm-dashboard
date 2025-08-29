from setuptools import setup, find_packages

setup(
    name="momac-sdm-dashboard",
    version="0.1.0",
    description="MOMAC Software-Defined Manufacturing Dashboard",
    author="MOMAC",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.25.0",
        "streamlit>=1.31.0",
        "pydantic>=2.6.0",
        "python-dateutil>=2.8.2",
        "openpyxl>=3.1.2",
    ],
)
