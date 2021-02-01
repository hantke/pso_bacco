from setuptools import setup 

setup(
    name="pso_bacco",
    author="Sergio Contreras Hantke",
    author_email="stcontre@uc.cl",
    description="Basic PSO",
    packages=['pso_bacco'],
    include_package_data=True,
    install_requires=["numpy","deepdish","mpi4py"],
)