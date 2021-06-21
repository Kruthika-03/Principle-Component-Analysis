# Principle-Component-Analysis
The Wine Dataset is used for classification using Principle Component Analysis. 

## Algorithm
1. Read the data set (d- dimensional)
2. Split it into training and testing set
3. Standardize the dataset
4. Construct the covariance matrix
5. Reduce the covariance matrix into: eigenvectors and eigenvalues
6. Sort the obtained eigenvalues in decreasing order
7. Select k-eigenvectors corresponding to k-largest eigenvalues, where k is the dimensionality of the new feature subspace (k ≤ d).
8. Next, construct projection matrix W from the “top” k eigenvectors.
9. Transform the d-dimensional input dataset X using the projection matrix W to obtain the new k-dimensional feature subspace.

## Dataset Description
I have considered the Wine Data Set from the UCI Machine Learning Repository. Here the wine is to be classified into three classes based on the chemical analysis which determines the origin of wines. The Data is multivariate, integer and real.
![dataset](https://user-images.githubusercontent.com/58825386/122722309-2b16c180-d28f-11eb-81aa-3b7715c1f4ab.png)
