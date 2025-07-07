# ibm_dclean1

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

