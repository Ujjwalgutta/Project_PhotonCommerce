# iReceipt Lens
iR Lens is an automated Receipt/Invoice Parser which extracts predefined entities from a document into structured data(key-value pairs). This project was developed to eliminate the inefficiencies caused due to manual data entry. This project has a wide range of commercial applications as well as daily personal uses.

## Motivation for this project:
- Every year there is a $120B worldwide due to inefficiencies related to manual data entry at fulfillment centres(Eg:Third party warehouses).
- Many Enterprises in the market have huge volumes of unstructured data and it keeps growing exponentially. Eg: Law firms, Warehouses, Banks etc.
- Track your monthly expenses without having to search for your receipts in a sea of documents. 
- Document Understanding using AI is a trending technology.

## Setup
Clone repository
```
repo_name=iR-Lens # URL of the project repository
username=Ujjwalgutta # Username 
git clone https://github.com/$username/$repo_name
```

## Requisites

- List all packages and software needed to build the environment
- This could include cloud command line tools (i.e. gsutil), package managers (i.e. conda), etc.

#### Dependencies

- [Streamlit](streamlit.io)

#### Installation
To install the package above, pleae run:
```shell
pip install -r requiremnts
```

## Build Environment
- Include instructions of how to launch scripts in the build subfolder
- Build scripts can include shell scripts or python setup.py files
- The purpose of these scripts is to build a standalone environment, for running the code in this repository
- The environment can be for local use, or for use in a cloud environment
- If using for a cloud environment, commands could include CLI tools from a cloud provider (i.e. gsutil from Google Cloud Platform)
```
# Example

# Step 1
# Step 2
```

## Configs
- We recommond using either .yaml or .txt for your config files, not .json
- **DO NOT STORE CREDENTIALS IN THE CONFIG DIRECTORY!!**
- If credentials are needed, use environment variables or HashiCorp's [Vault](https://www.vaultproject.io/)


## Test
- Include instructions for how to run all tests after the software is installed
```
# Example

# Step 1
# Step 2
```

## Run Inference
- Include instructions on how to run inference
- i.e. image classification on a single image for a CNN deep learning project
```
# Example

# Step 1
# Step 2
```

## Build Model
- Include instructions of how to build the model
- This can be done either locally or on the cloud
```
# Example

# Step 1
# Step 2
```

## Serve Model
- Include instructions of how to set up a REST or RPC endpoint
- This is for running remote inference via a custom model
```
# Example

# Step 1
# Step 2
```

## Analysis
- Include some form of EDA (exploratory data analysis)
- And/or include benchmarking of the model and results
```
# Example

# Step 1
# Step 2
```
