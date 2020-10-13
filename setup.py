from setuptools import setup 

with open("README.MD", "r") as readme_file:
    readme = readme_file.read()

setup(name = "iR-Lens",
        author = "Ujjwal Gutta",
        author_email = "ugutta@asu.edu",
        description = "Extract Information from Receipts and Invoices",
        long_description = readme
        url = "https://github.com/Ujjwalgutta/iR-Lens",
        packages = ['iReceipt_Lens'],
)
