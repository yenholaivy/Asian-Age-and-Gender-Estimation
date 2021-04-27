# Detect Gender and Age of Asian Individuals with Deep Learning

## Motivation
Is 'Asian Don't Raisin' a real thing? There are so many Asian celebrities in their 40's looking like they are in their 20's. Even myself, as an American of Taiwanese descent, I remember telling people I was a sophomore back in college, and they thought I meant high school the whole conversation. It seems to be a common thing that people can't guess Asian people's ages very accurately. That made me wonder if machines can do it better than human eyes; are there some facial features that we as human can't see easily but machines can pick up? I decided to build a model using Convolutional Neural Network (CNN) to detect the gender and age of Asian individuals!
<insert images>

## Data
The dataset used in this study is downloaded from [The Asian Face Age Dataset (AFAD)](https://afad-dataset.github.io/). The dataset includes more than 160,000 images with the age and gender well-labeled. There are 100,000 images of male and 60,000 images of female, and the age ranging from 15 to 41 with the average to be 25.6. The dataset has the most images from age 19 to 25, as the pictures were collected from a social media site. 
<insert images>
  
## Modeling
### Gender Detection
Before I created the age estimator model, I wanted to start with something easier -- to build a gender detection model (see this notebook). I used sklearn DummyClassifier as my baseline, and got a 53% accuracy. Then, I built a CNN classifier with two convolutional layers and two fully-connected layers. I trained the model for 50 epochs and got a 93% accuracy with male precision 96% and female precision 89%.
<insert image of the model>
<epoch image>
|  Model | Accurary | 
| --- | --- |
| DummyClassifier | 53% | 
| Simple CNN | 93% | 
<confusion matix>

Here are some examples of a male being detected as a female, and vice versa.
<insert image>
  
### Age Estimation
Next, I moved on to the age estimator model (see this notebook). My baseline is to always predict the average age 25.6 years, and it gives mean absolute error (MAE) of 5.2. My CNN model for age estimation has the same structure as the gender model, except the last layer has 'linear' as the activation function since this is a regression problem. I trained the model for 50 epochs and got a 4.1 MAE, which is 20% improvement from the baseline. 
<epoch image>
|  Model | MAE | 
| --- | --- |
| Baseline | 5.2 | 
| Simple CNN | 4.1 | 
  
When looking at the average MAE per age, I saw that the model was performing much better for the age range from 20 to 29 with average MAE of 2 to 4. However, the model struggled as the age deviated from the middle of the age range. Age 40 had the worst average MAE (over 10). This showed that the model was either optimizing the ages where it had the most images, or the model was just having a hard time picking up facial features for people below 20 and above 30.
<insert image>

#### Apply Transer Learning
To imporove the results, I decided to apply transfer learning (see this notebook). The pre-trained model I applied was VGGFace, which was a model already fine-tuned on faces (see article for details). This is a massive model with 38 layers and 145 millions parameters. Due to hardward limits of my EC2 instance, I removed the last few convolutional layers that had over 4000 filters, and added a few fully-connected layers.
<insert model summary>

On top of that, I decided to split up the genders into male and female and train two models. My thinking was, since the model could detect male and female with over 90% accuracy, there must be some noticable facial features that differentiated the two, and perhaps I could get better results having them separate. I recalculated the baseline MAEs for each gender and trained the two transfer learning models for 6 and 10 epochs, respectively. After over 60 hours, I got an exciting result with male MAE 3.6 (30% improvement from the baseline) and female also MAE 3.6 (25% improvement from the baseline).

<ephoc img>
  
 |  Model | MAE | 
| --- | --- |
| Baseline | 5.1 | 
| Male CNN | 3.6 |
  
<epoch image>
  
|  Model | MAE | 
| --- | --- |
| Baseline | 4.8 | 
| Female CNN | 3.6 | 

When looking at the average MAE per age, we can still see that both models were performing better for age 20 to 29. Interestingly, the female model performed better than the male model before age of 23, and the male model out-performed the female model after that. 
  

### Conclusion and Next Steps
In this study, I was able to use deep learning to create a gender detection model with 93% accuracy (40% improvement from the baseline) and two age estimation models for each gender with MAE 3.6 (30% improvement for male, 25% improvement for female). 
As for next steps, I would like to train the age estimation models for more epochs. Even though I was training the model on a g3 EC2 instance (3 GPUs, 16 CPUs and 100 RAM), it still took about 4 hours to finish one epoch.


## Final Notes
I wasn't able to upload the weights for my transfer learning models due to GitHub file size limit. If you are interested in using my pre-trained model, please feel free to contact me at yenholai.ivy@gmail.com.

## Tools
- AWS: S3, EC2
- Python: Tensorflow, Scikit-Learn, Numpy, Pandas, Matplotlib

## Citation
- Ordinal Regression With Multiple Output CNN for Age Estimation
Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, Gang Hua; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 4920-4928


