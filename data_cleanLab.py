import os
import warnings
warnings.filterwarnings("ignore")
import sys
print(sys.executable)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
# %matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.stats import norm
import requests

os.makedirs("plots/corrs", exist_ok=True)

def download(url, filename):
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)

path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data1.tsv"
download(path, "Ames_Housing_Data1.tsv")

housing = pd.read_csv("Ames_Housing_Data1.tsv", sep="\t")
head = housing.head(5)

# .info() is used to find info about fetures and types using info() method
housing_info = housing.info()

# .describe() to show the count, mean, min, max of sale price attribute
housing_sale_price = housing["SalePrice"].describe()
# .value_count shows information about categorical (object) attributes
housing_sale_condition = housing["Sale Condition"].value_counts()
print("===== Sale price ====")
print(housing_sale_price)
print("=== Sale Condition ===")
print(housing_sale_condition)


hous_num = housing.select_dtypes(include = ['float64', 'int64'])
hous_num_corr = hous_num.corr()['SalePrice'][:-1] #:-1 make latest row SalesPrice
#display pearsons correlation coefficient greater than 0.5
#Pearsons -  measures linear correlation between two sets of data
top_features = hous_num_corr[abs(hous_num_corr) > 0.5].sort_values(ascending=False)
print("=== Corr ===")
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(top_features), top_features))

for i in range(0, len(hous_num.columns), 5):
    graph = sns.pairplot(data=hous_num, 
                 x_vars=hous_num.columns[i:i+5], 
                 y_vars=['SalePrice'])
    file_name = f"plots/corrs/pairplot_{i//5 + 1}.png"
    graph.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()