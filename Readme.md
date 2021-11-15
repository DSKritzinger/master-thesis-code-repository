# Feature Selection in Gene Expression data for Disease Progression Prediction

The following code base contains the generation and evaluation scripts used to develop the Boruta-RFE hybrid feature selection method, aimed at the identification of minimal gene-signatures for TB progression prediction. 

## Components

The codebase consist of four main folders, i.e., exploratory data analysis, feature selection algorithms, evaluations and utils.

* ### Exploratory data analysis:

    The exploratory data analysis folder contains the R code used to explore the dataset used in this project. The code is split up based on the data base's explored, i.e., the ACS or GC6-74 datasets.

* ### Features selection algorithms: 

    The feature selection folder contains all the feature selection algorithms implemented in this project. These algorithms are seperated as either first phase or second phase algorithms, and the final Boruta-RFE implementation is added under "hybrid algorithms". 

    These algorithm implementations were specifically developed for the evaluation in this project, thus are not general purpose implementations.

* ### Utils:

    The utils folder contains the most general purpose utility functions used in this project, i.e., the cross-validation predictive performance generation function, the stability evaluation function, the median-ratio-standardization function, as well as the certain modules that were required to be locally edited and used.

* ### Evaluations:

    The evaluation folder contains notebook like scripts used for the evaluation of the feature selection algorithms, preprocessing methods and data characteristics for this project. This is seperated into first phase, second phase, data characteristic and test-validation set evaluations. As the evaluation procedures were relatively ad hoc, the scripts are also.

## Data and Results

The code base also contains the prepared data used for this project, as well as all generated results. Most results are in the form of "pickled" data, thus would have to be run in conjunction with the evaluations scripts to produce any visible results. 

## Running the scripts

All code in the repository can be run, but does require environment setup. For the python code, a requirements.txt file has been added which will install all the necessary dependencies required to run the scripts. For the R code, if using Rstudio, the necessary packages will be recommended automatically.


