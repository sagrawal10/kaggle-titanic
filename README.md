# kaggle-titanic
Solving the Kaggle Titanic problem using ensemble Learning and Neural Networks.

This project provides an analysis of what sorts of people were likely to survive during the titanic mishap. 

File structure:

/bin:
1. fetching_data.py: Fetching the data and removing outliers
2. graphical_analysis.ipynb: This is a python notebook that presents different graphical analysis.
3. feature_analysis.py: Based on the graphical analysis, data pruning and feature engineering is performed.
4. modelling_ensemble.py: In this, I have performed ensemble modelling with adaboost, Random Forest, Extra Trees and Gradient Boosting.
5. modelling_ann.py: In this, I have used a keras classifier

/bin-ann:
This folder deploys a keras model on data that is not processed extensively. Minimal feature analysis and engineering is performed on the train data.
And the accuracy is found to be slightly better than all methods mentioned previously.

/data:
train.py, test.py: Original data
train.pkl, test.pkl: Pickle files. Serialised dataframes after executing fetching_data.py.
train_final.pkl, test_final.pkl: Serialised dataframes after executing feature_analysis.py.
*.csv: Output files(used for submission).

How the code works?
You can run any of the .py files standalone as the required data files are provided using the pickle files.
However the correct sequence of execution is:
fetching_data.py -> feature_analysis.py -> modelling_*.py

Happy coding!