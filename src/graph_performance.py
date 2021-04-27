import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def graph_ACC(history,title):
    """
    Graph acc vs val_acc for the classifier.
    """
    _, ax = plt.subplots()
    ax.set_title(title)
    try:
        ax.plot(history.history['acc'], label='Train')
        ax.plot(history.history['val_acc'], label='Test')
    except:
        ax.plot(history['acc'], label='Train')
        ax.plot(history['val_acc'], label='Test')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()


def graph_MAE(history,title):
    """
    Graph loss vs val_loss for the regressor.
    """    
    _, ax = plt.subplots()
    ax.set_title(title)
    try:
        ax.plot(history.history['loss'], label='Train')
        ax.plot(history.history['val_loss'], label='Test')
    except:
        ax.plot(history['loss'], label='Train')
        ax.plot(history['val_loss'], label='Test')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Absolute Error")
    ax.legend()


def get_ytrue_ypred(model, datagen):
    """
    Helper funtion that generates a dataframe with y_pred, y_true and mae.
    """    
    y_true = np.array([])

    for i in range(len(datagen)):
        y_true = np.append(y_true, datagen[i][1])
        
    y_pred = model.predict(datagen)
    
    y_true1 = y_true + 15
    y_pred1 = (y_pred +15).reshape(-1)
    true_pred_df = pd.DataFrame({'y_true':y_true1, 'y_pred':y_pred1})
    true_pred_df['mae'] = np.abs(true_pred_df.y_true - true_pred_df.y_pred)
    
    return y_true1, y_pred1, true_pred_df


def get_average_MAE(true_pred_df):
    """
    Helper funtion that generates the average MAE for each age.
    """    
    age_group = true_pred_df.groupby('y_true')
    
    mae_average = []
    for age, age_data in age_group:
        mae_average.append(np.mean(age_data.mae))
        
    return mae_average


def get_count_train(datagen):
    """
    Helper funtion that generates the frequency of the training dataset.
    """
    y_train = np.array([])

    for i in range(len(datagen)):
        y_train = np.append(y_train, datagen[i][1])
        
    age_train, count_train = np.unique(y_train, return_counts=True)
    
    return count_train


def plot_average_MAE(train_datagen, val_datagen, model, gender = None):
    """
    Graph the average MAE for each age.
    """
    ages = np.arange(15,41)
    y_true, y_pred, true_pred_df = get_ytrue_ypred(model, val_datagen)
    mae_average = get_average_MAE(true_pred_df)
    count_train = get_count_train(train_datagen)
    print(count_train)
    
    fig, ax = plt.subplots(figsize = (12,5))
    ax.plot(ages, mae_average, label = 'Average MAE', linewidth = 2)
    ax.scatter(ages, mae_average)
    ax2 = ax.twinx()
    ax2.plot(ages, count_train, color = 'steelblue',label = 'Count of Images')
    ax2.fill_between(ages,count_train,alpha = 0.1, color='steelblue')
    ax2.set_ylabel('Count of Images per age')
    ax.set_xticks(ages)
    ax.set_xlabel('Age')
    ax.set_ylabel('MAE')
    ax.set_xlim(left=14.5, right=40.5)
    ax.set_ylim(bottom = 0)
    ax2.set_ylim(bottom = 0)
    ax2.grid(None)
    ax2.legend(loc = 'upper center')
    ax.legend()
    if gender == 'M':
        ax.set_title('Average MAE per age - Male')
    elif gender == 'F':
        ax.set_title('Average MAE per age - Female')
    else:
        ax.set_title('Average MAE per age')