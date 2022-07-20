# Code For The Project 
The code for this project is written in Python where each file tackles a different part of the process (and utils provides different auxiliary functions that are applicable to multiple files). 

## How To:
---

The code is currently mainly written in Python, although the starting point (with respect to inspiration, methodology, etc.) for this project in idea and approach was written in R. See Paxton in Useful links.

### To Install
To install the requisite libraries, once python (I am running Python 3.8.10) and pip, simply run 

    pip install -r requirements.txt

which will install all the modules at the versions that I currently have them installed.

### To Run

To run any file just enter (python or python3 depending on how it is setup on your computer)

    python file.py --arg1 arg1 --arg2 arg2 <etc.>

E.g.,

    python plot.py --month june -- rssi -90

Note: not currently only plotting has arguments required (will be changed in the future). So for the other files' functionality, modify as need be and just run python file.py, as with any other file

## Libraries:
---
The following are the libraries currently used as noted in the requirements.txt file:

* matplotlib==3.5.2
* numpy==1.22.4
* pandas==1.4.2
* scikit_learn==1.1.1
* scipy==1.8.1
* utm==0.7.0
* xgboost==1.6.1


## Useful Links:
---
This section will contain links to and possibly brief descriptions of useful artciles, stackoverflow pages, wikis, etc. 

### Multilateration
* https://github.com/lemmingapex/Trilateration
* https://github.com/jurasofish/multilateration
* https://stackoverflow.com/questions/17756617/finding-an-unknown-point-using-weighted-multilateration
* https://gis.stackexchange.com/questions/66/trilateration-using-3-latitude-longitude-points-and-3-distances
* https://stackoverflow.com/questions/8318113/multilateration-of-gps-coordinates
* https://github.com/glucee/Multilateration/blob/master/Python/example.py

### ML
* https://github.com/dmlc/xgboost/blob/master/demo/guide-python/multioutput_regression.py

