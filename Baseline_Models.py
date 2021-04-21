import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from EDA import text_to_pd, dist_by_gender

def gender_baseline_model(train, test):
    '''
    Gender baseline model. Always predict male.

    Parameter:
        train, test

    Return:
        gender_baseline
    '''
    y_true_gender = test.gender.to_numpy()
    y_pred_gender = np.ones(y_true_gender.shape)
    gender_baseline = round(accuracy_score(y_true_gender, y_pred_gender),4)
    print(f'Accuracy for the baseline model for gender is : {gender_baseline}')
    return gender_baseline

def age_baseline_model(train, test):
    '''
    Age baseline model. Always predict the mean age.

    Parameter:
        train, test

    Return:
        age_baseline
    '''
    mean_age = train.age.mean()
    y_true_age = test.age.to_numpy()
    y_pred_age = np.full(y_true_age.shape, mean_age)
    age_baseline = round(mean_absolute_error(y_true_age, y_pred_age),2)
    print(f'Mean Age is: {mean_age}')
    print(f'MAE for the baseline model for age is: {age_baseline}')
    return age_baseline


if __name__ == '__main__':
    file_path = 'AFAD-Lite.txt' # Use the provided text file from AFAD
    data = text_to_pd(file_path)
    train, test = train_test_split(data, test_size=0.2)
    gender_baseline = gender_baseline_model(train, test)
    age_baseline = age_baseline_model(train, test)