import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def text_to_pd(file_path):
    '''
    Convert the AFAD text file into a pandas dataframe with additional columns of age and gender.

    Parameter:
        file_path (str): The file path to the text file.

    Return:
        data (pd.DataFrame):  Pandas dataframe with additional columns of age and gender.
    '''

    data = pd.read_csv(file_path,header=None)
    data['age'] = data[0].apply(lambda x: int(str(x)[2:4])) # Extract age
    data['gender'] = data[0].apply(lambda x: 0 if str(x)[5:8] == '112' else 1) # Extract gender, 1 = Male, 2 = Female
    data = data[data['age'] < 41] # Only include age < 41

    age_group = {15: '15-17',  16: '15-17', 17: '15-17', 18: '18-20', 19: '18-20', 20: '18-20',
     21: '21-23', 22: '21-23', 23: '21-23', 24: '24-26', 25: '24-26', 26: '24-26',
     27: '27-29', 28: '27-29', 29: '27-29', 30: '30-32', 31: '30-32', 32: '30-32',
     33: '33-35', 34: '33-35', 35: '33-35', 36: '36-40', 37: '36-40', 38: '36-40',
     39: '36-40', 40: '36-40'}

    data['age_group'] = data['age']
    data.replace({'age_group': age_group}, inplace = True)

    return data

def dist_by_gender(data):
    '''
    Graph the number of images by age and gender.

    Parameter:
        data (pd.DataFrame): Pandas dataframe with age and gender.

    Return:
        None
    '''
    data_by_gender = data.groupby('gender')

    for gender, gender_data in data_by_gender:
        if gender == 0:
            female_dist = gender_data.age.value_counts().sort_index()
        else:
            male_dist = gender_data.age.value_counts().sort_index()

    fig, ax = plt.subplots(figsize = (12, 5))

    ax.bar(male_dist.index-0.15, male_dist.values, width = 0.3,color= 'steelblue', label = 'Male')
    ax.bar(female_dist.index+0.15, female_dist.values, width = 0.3, color= 'firebrick', label = 'Female')

    total_mean = round(data.age.mean(),2)
    ax.axvline(x = total_mean, c ='orange', lw = 3, linestyle = '--', label = 'Average Age')
    
    ax.set_xticks(female_dist.index)
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of Images')
    ax.set_title('Number of Images by Age and Gender')
    ax.legend()
    fig.tight_layout()
    plt.show()

    

