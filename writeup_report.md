#**Behavioral Cloning**

##Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./writeup_images/center.jpg "Center"
[image3]: ./writeup_images/recovery1.jpg "Recovery Image"
[image4]: ./writeup_images/recovery2.jpg "Recovery Image"
[image5]: ./writeup_images/recovery3.jpg "Recovery Image"
[image6]: ./writeup_images/normal.jpg "Normal Image"
[image7]: ./writeup_images/flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.   

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model uses the architecture proposed by nvidia on their paper http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 66-75)

The model includes RELU layers to introduce nonlinearity on every convolutional layer, and the data is normalized in the model using a Keras lambda layer (code line 64).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 78).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also driving in zigzag. Sometimes I also re-recorded conflicting parts of the track like the bridge, the dirt or the ramps.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to record driving data and use convolutional layers to process the images and then let the simulator drive itself and see if it improves the previous setup.

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because it has few convolutional layers and few fully conncted layers and it is useful for workign with images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that I followed the nvidia architecture that has more convoluional layers with bigger filters and added dropout.

Then I recorded the first track in the oposite direction and I also recorded track 2 few times to generalize better. I also preprocessed the images by adding a mirrored copy of the dataset with the negative steering angle and I also added the images from the side cameras with a small steeting angle correction.(model.py lines 29-48)

The final step was to run the simulator to see how well the car was driving around track one. I noticed that my car was going from side to side when entering curves with the red and white strips so to improve those areas I recorded more samples of those curves. Then I also noticed that it had problems with the dirt part so I recorded that part again. Later my car was trying to avoid the shadows from the trees so I also recorded that part.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 62-77) consisted of a convolution neural network with the following layers and layer sizes:
Convolutional 24x5x5
Convolutional 36x5x5
Convolutional 48x5x5
Convolutional 64x3x3
Convolutional 64x3x3
Fully connecter 1064
Fully connected 100
Dropout 0.1
Fully connected 50
Dropout 0.1
Fully connected 10
Dropout 0.1
Fully connected 1


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help avoid left bias as the track1 is couterclockwise. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

I also use left and right camera images with a correction factor of +/-1 to the steering angle and add the normal and the flipped versions of them to the dataset.

After the collection process, I had 16399 data ponints. I then preprocessed this data by cropping the images 70 pixels from the top and 20 pixels from the bottom using a cropping layer function (train.py line 63) to avoid the background and the car hood interfering with the training.
Then I also normalize the images with a lambda function (train.py line 64)


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the validation error. I used an adam optimizer so that manually training the learning rate wasn't necessary.
