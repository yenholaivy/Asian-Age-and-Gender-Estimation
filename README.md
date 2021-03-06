# Detect Gender and Age of Asian Individuals with Deep Learning

## Table of Contents
- [Motivation](#motivation)
- [Data](#data)
- [Modeling](#modeling)
    - [Gender Detection](#gender-detection)
    - [Age Estimation](#age-estimation)
    - [Transfer Learning](#transer-learning)
- [Conclusion and Next Steps](#conclusion-and-next-steps)
- [Final Notes](#final-notes)
- [Tools](#tools)
- [Citation](#citation)


## Motivation
Is 'Asian Don't Raisin' a real thing? There are so many Asian celebrities in their 40's looking like they are in their 20's. Even myself, as an American of Taiwanese descent, I remember telling people I was a sophomore back in college, and they thought I meant high school the whole conversation. It seems to be a common thing that people can't guess Asian people's ages very accurately. That made me wonder if machines can do it better than human eyes; are there some facial features that we as humans can't see easily but machines can pick up? I decided to build a model using Convolutional Neural Network (CNN) to detect the gender and age of Asian individuals!

![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/asian_females.png)


## Data
The dataset used in this study was downloaded from [The Asian Face Age Dataset (AFAD)](https://afad-dataset.github.io/). The dataset includes more than 160,000 images with the age and gender well-labeled. There are 100,000 images of male and 60,000 images of female, and the ages ranging from 15 to 41 with the average of 25.6. The dataset has the most images from age 19 to 25, as the pictures were collected from a social media site. 

![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/afad_samples.png)

![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/dist_age_gender.png)
  
  
## Modeling
### Gender Detection
Before I created the age estimator model, I wanted to start with something easier -- to build a gender detection model (see [simple_cnn_gender_detection.ipynb](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/simple_cnn_gender_detection.ipynb)). I used sklearn DummyClassifier as my baseline, and got a 53% accuracy. Then, I built a CNN classifier with two convolutional layers and two fully-connected layers. 

![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/simple_cnn_struc.png)

I trained the model for 50 epochs and got a 93% accuracy with male precision 96% and female precision 89%.

|  Model | Accurary | 
| --- | --- |
| DummyClassifier | 53% | 
| Simple CNN | 93% |

![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/simple_cnn_gender.png)

|  Gender | Precision | 
| --- | --- |
| Male | 96% | 
| Female | 89% |

![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/cm_gender.png)

Here are some examples of a male being predicted as female, and vice versa.

![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/wrong_predictions.png)
  

### Age Estimation
Next, I moved on to the age estimator model (see [simple_cnn_age_estimator.ipynb](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/simple_cnn_age_estimator.ipynb)). My baseline is to always predict the average age 25.6 years, and it gives mean absolute error (MAE) of 5.2. My CNN model for age estimation has the same structure as the gender model, except the last layer has 'linear' as the activation function since this is a regression problem. I trained the model for 50 epochs and got a MAE of 4.1, which is a 20% improvement compared to the baseline. 

|  Model | MAE | 
| --- | --- |
| Baseline | 5.2 | 
| Simple CNN | 4.1 | 

![alt img](https://user-images.githubusercontent.com/77142026/116288696-9f792c80-a746-11eb-93f5-2cf3429ce7a5.png)
  
When looking at the average MAE per age, I saw that the model was performing much better for the ages from 20 to 29 with average MAE of 2 to 4. However, the model struggled as the age deviated from the middle of the age range. Age 40 had the worst average MAE (over 10). This showed that the model was either optimizing the ages where it had the most images, or the model was just having a hard time picking up facial features for people below 20 and above 30.

![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/simple_cnn_mae.png)


#### Transer Learning
To improve the results, I decided to apply transfer learning (see [transfer_learning.ipynb](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/transfer_learning.ipynb)). The pre-trained model I applied was VGGFace, which was a model already fine-tuned on faces (see [article](https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/) for details). This is a massive model with 38 layers and 145 millions parameters. Due to hardward limits of my EC2 instance, I removed the last few convolutional layers that had over 4000 filters, and added a few fully-connected layers.

![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/transfer_learning.png)

On top of that, I decided to split up the genders into male and female and train two models. My thinking was, since the model could detect male and female with over 90% accuracy, there must be some noticable facial features that differentiated the two, and perhaps I could get better results having them separate. I recalculated the baseline MAEs for each gender and trained the two transfer learning models for 6 and 10 epochs, respectively. After over 60 hours, I got an exciting result with male MAE 3.6 (30% improvement from the baseline) and female also MAE 3.6 (25% improvement from the baseline). Even though the loss for the male model was still decreasing, I had to stop the training due to time constraints.
  
 |  Model | MAE | 
| --- | --- |
| Baseline | 5.1 | 
| Male CNN | 3.6 |

![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/vgg_male.png)
  
|  Model | MAE | 
| --- | --- |
| Baseline | 4.8 | 
| Female CNN | 3.6 | 

![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/vgg_female.png)

When looking at the average MAE per age, we can see the purple line (i.e female model), performed the best at age 20 and 21, but the model struggled more and more after the age of 30. Comparing to the green line (i.e male model), the model performed best at the age of 23, overall no MAE over 10, which is quite impressive given that I was only able to train it for 6 epochs.
  
![alt img](https://github.com/yenholaivy/Asian-Age-and-Gender-Estimation/blob/main/img/vgg_m_vs_f.png)


## Conclusion and Next Steps
In this study, I was able to use deep learning to create a gender detection model with 93% accuracy (40% improvement from the baseline) and two age estimation models for each gender with a MAE of 3.6 (30% improvement for male, 25% improvement for female). 

As for next steps, I would like to continue improving the models. To make the model predicting all ages more accurately, I would like to collect more images for ages below 20 and above 30. Also, I would like to train the transfer models for more epochs. Even though I was training the model on a g3 EC2 instance (3 GPUs, 16 CPUs and 100 RAM), it still took about 4.5 hours to finish one epoch. Since the test MAE was still decreasing (see above image), I believe the model would continue to improve with more epochs.


## Final Notes
I wasn't able to upload the weights for my transfer learning models due to GitHub file size limit. If you are interested in using my pre-trained model, please feel free to contact me at yenholai.ivy@gmail.com.

## Tools
- Cloud Computing: AWS S3, AWS EC2
- ML/Backend: Tensorflow Keras, Scikit-Learn, Numpy, Pandas
- Data Visualization: Matplotlib, Seaborn

## Citation
- Ordinal Regression With Multiple Output CNN for Age Estimation
Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, Gang Hua; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4920-4928


