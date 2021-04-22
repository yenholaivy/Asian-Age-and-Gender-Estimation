import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def graph_MAE(history, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(history.history['loss'], label='Train')
    ax.plot(history.history['val_loss'], label='Test')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MAE")
    ax.legend()


def graph_ACC(history,title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(history.history['acc'], label='Train')
    ax.plot(history.history['val_acc'], label='Test')
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.legend()