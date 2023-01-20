![image](https://user-images.githubusercontent.com/113465005/213256471-04917667-69ab-4126-8cc6-72bc940f0fe0.png)


# Retail Recommender Solution Offering
The Retail Recommender Solution Offering unlocks data for clients to provide specific recommendations across item features such as; products, product categorites, sellers, customers, reviews on products, and more.

![Link to Introductory Solution Workshop Deck](https://microsoft-my.sharepoint.com/:p:/p/jheseltine/EaoNnJooT59Ehr8jnagJkgQBo3NTZnB4CnZLcQDCwKsAPA?e=Y281Vj)

# Prerequisites
* Azure subscription
* Azure Machine Learning dedicated workspace
* *Optional* Databricks workspace

**Prerequisites Client 2 Day Workshop**
* Microsoft members need to be added as Guests in clients Azure AD
* A Resource Group (RG)  needs to be set for this Workshop POC, in the customer Azure tenant
* The customer team and the Microsoft team must have Contributor permissions to this resource group
* A storage account must be set in place in the RG
* Datasets must be uploaded as CSV or Parquet files to the blob storage account, at least one week prior to the workshop date
* Azure Machine Learning Workspace must be deployed in the RG
* Optional but recommended – Databricks Workspace deployed in the RG


# The Benefits
* In comparison with ![this MSFT accelerator](https://github.com/microsoft/Azure-Synapse-Retail-Recommender-Solution-Accelerator), our Retail Recommender Solution Offering is much simpler to setup. It doesn't require Synapse or Spark while at the same time can handle more than 95% of the retail cases
* Uses not only the transaction log (Date, User_id, Item_id, Interaction), but can also use the Item master dataset and the user dataset in order to make quality predictions 
* Leverages side features for training and for prediction. Can take in consideration side features for transactions , users and items.
* Can perform item similarity / basket analysis for upsell and cross sell
* Produces as output an API on a docker image. That can be place in any cloud provider or on-premises
* Contains deployment via Azure ML Services in:  Azure Container Instance (Test), and Azure Kubernetes Service (Production)}
* Also contains deployment for Azure Databricks MLFlow model serving
* API Features:  Top K recommendations, include or exclude set of items from the recommendation,  uses side features at prediction time, specify specific features at inference time,  allows querying similar items.

# Getting Started and Process Overview 
1. Create a compute instance within your Azure Machine Learning workspace (can use Standard DS11_v2)
2. Once your compute intance is created, launch your JupyterLab from your compute instance 
3. Open the terminal within your JuypterLab and clone this repo:
```
# enter your own user directory below
cd /Users/pabmar/

# clone repo
git clone https://github.com/pablomarin/Retail-Recommender.git

# change directory to retail recommender
cd Retail-Recommender/
```
4. 

# From repo to POC with your Client in 2 Days
* Add in workshop details
* link powerpoint here

# Azure and Analytics Platform
The directions provided for this repository assume fundemental working knowledge of Azure Machine Learning Service and Azure Databricks MLFlow.

For additional training and support, please see:
1. Azure Machine Learning Services
2. Azure Databricks MLFlow

