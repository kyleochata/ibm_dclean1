# ibm_dclean1

Practice data anlaysis and data cleaning (normalization / standardization). This is using housing data from IBM. Goals are to use log functions to transform the data, handle duplicate data entries, handle missing values, standardize & normalize data, and handle outliers. 

Packages needed are:
- pandas for managing data
- numpy for mathematical operations
- seaborn for data visualization
- matplotlib for visualizing the data
- sklearn for machine learning and machine-learning-pipeline related functions
- scipy for statistical computations

## Set up venv
Create the virtual environment to download needed packages
```bash
    python3 -m venv venv
```
Activate environment:

MAC/Linux
``` 
    source venv/bin/activate 
```
Windows(CMD)
```
    .\venv\Scripts\activate.bat
```
Windows(PowerShell)
```
    .\venv\Scripts\Activate
```

Install packages:
```
    pip isntall pandas numpy seaborn matplotlib scikit-learn scipy
```

To ensure that the packages are installed run `pip list` in the terminal with the venv active

**Known issue**:
    Be sure to change the interpreter to the venv corresponding interpreter to recognize the downloaded packages

