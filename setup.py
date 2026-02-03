from setuptools import setup 

setup(
    name="pso_bacco",
    author="Sergio Contreras Hantke",
    author_email="scontreras1@us.es",
    description="Basic PSO",
    packages=['pso_bacco'],
    include_package_data=True,
    install_requires=["numpy","mpi4py"],
)
