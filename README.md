# BayesianClimate
This repository contains some code that conducts a Bayesian analysis of climate data. Additionally there is a presentation about the results of the analysis.

In more detail, the climate data of the world since 1900 (no precise data exists for earlier dates) has been considered. 
Three different models has been used to analyse this date: 
  1. Bayesian Linear Regressin.
  2. Bayesian Polynomial Regression.
  3. A linear regression model with two break points.
  
The different models have been implemented in Python and for each of them three different methods have been used to calculate the posterior distribution, these are
  1. Laplace Approximation
  2. Gaussian Variational Approximation
  3. Metropolis Hastings Algorithm

The scripts can be changed easily so that the analysis is not conducted for the whole world but for a specific country.
There is a presentation which briefly analyses the findings of the analysis and compares the different models and methods.

Remark:
The script was developed with python 3.5 and any version above should work as well - backwards compatibility is not guaranteed though. I recommend creating a new python virtual environment and installing the required dependencies from the requirements.txt file by running the following command from within the KatzenAnalyser\ directory:
pip install -r requirements.txt
