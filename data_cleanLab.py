import os
import warnings
warnings.filterwarnings("ignore")
import sys
print(sys.executable)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
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
print(f"There is {len(top_features)} strongly correlated values with SalePrice:\n{top_features}")

for i in range(0, len(hous_num.columns), 5):
    graph = sns.pairplot(data=hous_num, 
                 x_vars=hous_num.columns[i:i+5], 
                 y_vars=['SalePrice'])
    file_name = f"plots/corrs/pairplot_{i//5 + 1}.png"
    graph.savefig(file_name, dpi=300, bbox_inches='tight')
    plt.close()

#inspect if 'SalePrice' data is normally distributed.
#must be normally distributed if any type of regression analysis to be performwed

# visual method of determining normal distribution through displot() in seaborn lib
sp_untransformed = sns.displot(housing['SalePrice'])
os.makedirs("plots/distribution", exist_ok=True)
fname = f"plots/distribution/untransformed.png"
sp_untransformed.savefig(fname, dpi=300, bbox_inches="tight")
plt.close()

# current plot shows a positive (left skew). use .skew() method on housing to calc skewness
print(f"Skewness_sp_untran: {housing['SalePrice'].skew()}") #1.743222; -0.5 - 0.5 is symmetrical skew; moderate skew -0.5 to -1; high = > 1 || < -1

# transform data to make is more normally distributed .log => natural log function
log_transformed = np.log(housing['SalePrice'])
sp_transformed = sns.displot(log_transformed)
fname = f"plots/distribution/transformed_natLog.png"
sp_transformed.savefig(fname, dpi=300, bbox_inches="tight")
plt.close()

print(f"Skewness_transformed: {log_transformed.skew()}") #-0.015 normal symm skewness

# inspection of 'Lot Area' feature. Positive (left) skew
la_untransformed = sns.displot(housing['Lot Area'])
fname = f"plots/distribution/lot_area_untran.png"
la_untransformed.savefig(fname, dpi=300, bbox_inches="tight")
plt.close()
print(f"Skewness_la_untran: {housing['Lot Area'].skew()}") # 12.778

# transform lot area data to be more normally distributed
la_log_transformed = np.log(housing['Lot Area'])
la_transformed = sns.displot(la_log_transformed)
fname = f"plots/distribution/lot_area_trans.png"
la_transformed.savefig(fname, dpi=300, bbox_inches="tight")
plt.close()
print(f"Skewness_la_trans: {la_log_transformed.skew()}") # -0.494

# Duplicate Handling via pandas duplicated() by 'PID' column
duplicate = housing[housing.duplicated(['PID'])]
print(f"Duplicates: \n{duplicate}")

# Remove duplicate
duplicate_remove = housing.drop_duplicates()
print(duplicate_remove)
# check for duplicated indexes 
print(housing.index.is_unique)
# remove duplicates on specific columns
rmv_sub = housing.drop_duplicates(subset=['Order'])
print(rmv_sub)

# Handle missing Vals
# isnull() - summarize all missing data values
total = housing.isnull().sum().sort_values(ascending=False)
# plot first 20 columns using matplotlib
total_select = total.head(20)
total_select.plot(kind="bar", figsize = (8,6), fontsize=10)
plt.xlabel("columns", fontsize=20)
plt.ylabel("count", fontsize=20)
plt.title("Total Missing Values", fontsize=20)
plt.savefig('plots/missing_values_plot.png', dpi=300, bbox_inches="tight")
plt.close()

# dropna() - all rows with a null value for 'Lot Frontage' will be dropped
print('dropna')
print(housing.dropna(subset=["Lot Frontage"])) #489 rows dropped due to null in "Lot Frontage"

# drop() - drop the whole column that contains missing values; entire column containing the null values dropped
print('drop')
print(housing.drop("Lot Frontage", axis=1)) # keeps the entire number of rows, just lose the col

# fillna() - replace the missing values
median = housing["Lot Frontage"].median()
print(f"median: {median}")
housing["Lot Frontage"].fillna(median, inplace=True)
print(housing.tail()) #indx 2927 has the median filled in for lot frontage

mva_is_na = housing["Mas Vnr Area"].isna()
mean = housing["Mas Vnr Area"].mean()
print(f"Mean: {mean}")
housing["Mas Vnr Area"].fillna(mean, inplace=True)
upd_rows = housing.loc[mva_is_na, "Mas Vnr Area"]
print(f"{len(upd_rows)} have been changed to {mean} for Mas Vnr Area")

# Feature Scaling
# min-max scaling (normalization) = simplest: values are shifted and rescaled so they end up ranging from 0 to 1
#Sub min value and divide by (max - min)
# Standardization - subtract mean value (standardized values always have a zero mean), then divided by std dev
# results in distribution with unit variance
# MinMaxScaler & StandardScaler

# Normalize data
norm_data = MinMaxScaler().fit_transform(hous_num)
print(norm_data)



# Standardize data
std_data = StandardScaler().fit_transform(hous_num)
print(std_data)

# standardize single column. StdScaler demans a 2D input; our sp column is 1D. 
sp = housing[["SalePrice"]]
std_sale_price = StandardScaler().fit_transform(sp)
# housing["SalePrice_Standardized"] = std_sale_price.flatten() # converts the 2D output back to 1D
print(std_sale_price[:5])

# Manual Standardization
sp_mean = housing["SalePrice"].mean()
sp_stdDev = housing["SalePrice"].std()

housing["SalePrice_Standardized"] = (housing["SalePrice"] - sp_mean) / sp_stdDev

print(housing["SalePrice_Standardized"].head(5))

# Handle Outliers

# finding outlier - Uni-variate analysis or Multi-Variate analysis (one var vs multiple vars)
# Uni-Variate Analysis
os.makedirs('plots/outliers', exist_ok=True)
sns.boxplot(x=housing["Lot Area"])
plt.savefig('plots/outliers/lot_area_boxplot.png', dpi=300, bbox_inches="tight")
plt.close()
sns.boxplot(x=housing['SalePrice'])
plt.savefig('plots/outliers/sale_price_boxplot.png', dpi=300, bbox_inches="tight")
plt.close()

# Bi-Variate Analysis
price_area = housing.plot.scatter(x="Gr Liv Area", y="SalePrice")
plt.savefig('plots/outliers/grLivArea_vs_SalePrice.png', dpi=300, bbox_inches="tight")
plt.close()

# Deleting Outliers
housing.sort_values(by="Gr Liv Area", ascending=False)[:2]
outliers_dropped = housing.drop(housing.index[[1499, 2181]])
new_plot = outliers_dropped.plot.scatter(x="Gr Liv Area", y="SalePrice")
plt.savefig('plots/outliers/dropOut_gla_vs_sp', dpi=300, bbox_inches="tight")
plt.close()

#Outlier handling in "lot area"
sns.boxplot(x=housing['Lot Area'])
plt.savefig('plots/outliers/la_boxplot.png', dpi=300, bbox_inches="tight")
plt.close()
price_lot = housing.plot.scatter(x="Lot Area", y="SalePrice")
plt.title("SalePrice vs Lot Area (Raw)")
plt.savefig('plots/outliers/price_lot_scatter.png', dpi=300, bbox_inches="tight")
plt.close()
housing['Lot_Area_Stats'] = stats.zscore(housing["Lot Area"])
print(housing[["Lot Area", "Lot_Area_Stats"]].describe().round(3))

# Remove any outliers where they are > 3 std deviations away from the mean
outliers = housing[abs(housing["Lot_Area_Stats"]) > 3] 
outlier_indices = outliers.index.tolist()
if outlier_indices:
    lot_area_remove = housing.drop(outlier_indices)
    print(f"removed outliers at indicies: {outlier_indices}")
else:
    lot_area_remove = housing.copy()
    print("No outliers found")
# lot_area_remove = housing.drop(housing.index[957])
lot_area_remove.plot.scatter(x="Lot Area", y="SalePrice")
plt.title("SalePrice vs Lot Area (Outliers Removed)")
plt.savefig('plots/outliers/la_price_dropped.png', dpi=300, bbox_inches="tight")
plt.close()

# Z-Score Analysis - identify Outliers mathematically. > 3 || < -3 means outliers generally

housing["LQFSF_Stats"] = stats.zscore(housing["Low Qual Fin SF"])
print(housing[["Low Qual Fin SF", "LQFSF_Stats"]].describe().round(3))
