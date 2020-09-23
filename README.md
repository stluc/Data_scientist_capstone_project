# Udacity Data Scientist Capstone Project
## Arvato project about prediction on new customers for a mail handout campaign. 

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

- SciKit-Learn
- PyTorch
- imblearn
- Matplotlib
- numpy
- Pandas
- hvPlot
- PyTorch
- Jupyter Notebook
- The code should run with no issues using Python versions 3.* (Python 3.8 was used).

## Project Motivation<a name="motivation"></a>
The task is to correctly predict new possible customers for a mail handout marketing campaign for the company Arvato, which offers products by mail order. This mostly comes up to a binary classification problem.

The data provided was heavily imbalanced and a lot of data from the features was missing. The project assigned as capstone project by Udacity was a good exercise to handle data which must be interpreted and sorted out as well as understanding the issues regarding machine learning models during data science process.

## File Descriptions <a name="files"></a>

The full analysis is contained in the jupyter notebook. Most of the functions are written in a separate file that needs to be imported named 'aid_functions.py'.
Due to privacy agreement the dataset used is not made available as well as all the models and data interpretation sheets.


## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@iamjuststen/customer-prediction-in-imbalanced-dataset-the-arvato-case-513822cdbbfe).
The model scored 0.7399 (AUC) in the [Kaggle competition](http://www.kaggle.com/t/21e6d45d4c574c7fa2d868f0e8c83140), but there are still high margins for improval.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit to Arvato and Udacity for providing the Data and the challenge.
