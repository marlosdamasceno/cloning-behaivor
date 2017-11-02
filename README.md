[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./Resources/ArchitectureNvidia.png "Nvidia's architecture"
[image2]: ./Resources/LossGraph.png "Training Loss View"
[image3]: ./Resources/ImageData001.jpg "Center Image Correct Direction"
[image4]: ./Resources/ImageData002.jpg "Left Image Correct Direction"
[image5]: ./Resources/ImageData003.jpg "Right Image Correct Direction"
[image6]: ./Resources/ImageData004.jpg "Center Image Opostie Direction"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Required Files

#### 1. The submission includes a model.py file, drive.py, model.h5 a writeup report and video.mp4.

My project includes the following files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network
* **video.mp4** with the output of the result in autonomous mode
* **Readme.md** summarizing the results

### Quality of Code

#### 1. The model provided can be used to successfully operate the simulation.
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
You may set the speed to maximum 20 mph, after that the car will leave the road.

#### 2. The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed.

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works =)

### Model Architecture and Training Strategy

#### 1. The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model.

The architecture of the model follows the Nvidia neural network, with the addition of two Dropouts layers.
1. As input layer a Keras lambda layer to normalize the data.
2. A cropping layer to reduce the image and reduce "noise" information.
3. Five convolution layers, two 5x5 and tree 3x3. All of them with Relu activation, to introduce nonlinearity. Depths are between 24 and 64.
4. One flatten layer.
5. A fully connected layer with 100 as output, with Relu activation.
6. A dropout layer, with 0.5 rate.
7. A fully connected layer with 50 as output, with Relu activation.
8. To finish two more fully connected layers, one with 10 and the last one with 1 output. No activation function for them.

Here is a image of the architecture.
![alt text][image1]

#### 2. Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce over-fitting.

The model contains two dropout layers in order to reduce over-fitting. Moreover, the model was trained and validated on different data sets to ensure that the model was not over-fitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Learning rate parameters are chosen with explanation, or an Adam optimizer is used.

The model used an Adam optimizer, so the learning rate was not tuned manually.

####4. Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track).

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving two laps at total. Plus on lane driving in the side of the road. Plus two more laps counter-clockwise. Totalizing 14.049 images in the training and validation set. I did not used the flip data augmentation, it was not needed.

### Architecture and Training Documentation

#### 1. The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem.

The overall strategy for deriving a model architecture (AlexLenet) was to first get some data and see whether the model was working or not. Therefore, I did one lap and run 10 epochs cheeking if the validation loss was decreasing.
After that, I collected more data following the tips of the class.
Trained again, however the results was not that good. Therefore, I changed the architecture to Nvidia as it was in the class and added two dropouts to avoid over-fitting. After 15 epochs I got a good result, I trained until 30 epochs just to see the result of it, however it was just over-fitting. There is no need to train after 15 epochs, that gives a great result, the validation loss is about 0.01. To train this network it took 700 seconds.
Here is a graph of the training.

![alt text][image2]

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

The architecture is the Nvidia one plus two dropouts. Please, see above on item Model Architecture and Training Strategy for more details.

#### 3. The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included.

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded the vehicle recovering from the sides of the road back to center so that the vehicle would learn to keep in the road even in a odd condition:

![alt text][image4]  ![alt text][image5]

Moreover, I did two additional laps on the counter-clockwise.

![alt text][image6]


### Simulation
#### 1. No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle).
Yes the vehicle was all the time on the road, with a speed of 9mph, as is possible to see on the video.mp4 file on the folder Resources.
I will add more three links to Youtube videos where is possible to see the car in a higher speed, 20mph.
1) [Car at speed of 9mph and showed from outiside](https://youtu.be/jq825Z38ApM) (this is the same result as video.mp4, however, on another perspctive of view)
2) [Car at speed of 20mph and showed from iniside](https://youtu.be/6zT51eMjZh4)
3) [Car at speed of 20mph and showed from outiside](https://youtu.be/v3JjXCj78CU)
