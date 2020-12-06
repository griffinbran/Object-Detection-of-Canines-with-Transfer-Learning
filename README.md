#  Machines Best Friend

# ![](https://todaysveterinarypractice.com/wp-content/uploads/sites/4/2019/01/Chocolate-Lab_shutterstock_405052057_Paul-Ekert.jpg)

Capstone Project

---
### Problem Statement
***Can an image be used to accurately describe itself?***
* Visual Question Answering (VQA)
* Audience:  Computer vision enthusiasts, dog lovers, security services, and the visually impaired
* Image data is a rich source of information. This project will aims to  automate the task of extracting  image descriptions.


  Questions to be explored:
> 1. Is the dog inside or outside?
> 2. Does it have a friend?
> 3. What breed is it?
> 2. What layers need to be pre-trained?
> 3. What is a reasonable 'optical cue'?
<br>
---
### Overview

This DSI module covers:

- Machine Learning for Deep Neural Networks (TensorFlow, Keras API)
- 
- Computer Vision ( RGB image processing, image formation, feature detection, computational photography)
- Convolutional Neural Networks(CNN)- regularization, automated pattern recognition, ...
- Transfer Learning with a pre-trained deep learning image classifier (VGG-16 CNN from Visual Geometry Group in 2014)

### Contents

* [Background](#background)
* [Data Aquisition & Cleaning](#data_aquisition_and_cleaning)
* [Exploratory Analysis](#exploratory_analysis)
* [Findings and Recommendations](#findings_and_recommendations)
* [Next Steps](#next_steps)
* [Software Requirements](#software_requirements)
* [Acknowledgements and Contact](#acknowledgements_and_contact)
---
<a id='background'></a>
### Background

Here is some background info:
> * Transfer learning: pre-existing model, trained on a LOT of data, used elsewhere.
> * Eliminates the need to afford cost of training deep learning models from scratch
> * Deep CNN model training short-cut, re-use model weights from pre-trained models previously developed for benchmark tests in comupter vision
> * VGG, Inception, ResNet: 
> * Weight initialization: weights in re-used layers used as starting point in training and adapted in response to new problem
> 1. Use model as-is to classify new photographs
> 2. Use as feature extraction model, output of pre-trained from a layer prior to output layer used as input to new classifier model
> *  Tasks more similar to the original training might rely on output from layers deep in the model such as the 2nd to last fully connected layer
> * Layers learn:
> 1. Layers closer to the input layer of the model:
    Learn low-level features such as lines, etc.
> 2. Layers in the middle of the network of layers:
    Learn complex abstract features that combine the extracted lower-level features from the input
> 3. Layers closer to the output:
    Interpret the extracted features in the context of a classification task
> * Fine-tuning learning rate of pre-trained model 
> * Transfer Learning Tasks
> 1.
> 2. 
> 3.
> 4.
> * Architectures:
> 1. Consistent and repeating structures (VGG)
> 2. Inception modules (GoogLeNet)
> 3. Residual modules  (ResNet)


### Data Dictionary

**NOTE: Make sure you cross-reference your data with your data sources to eliminate any data collection or data entry issues.**<br>
*See [Acknowledgements and Contact](#acknowledgements_and_contact) section for starter code resources*<br>

|Feature|Type|Dataset|Category|Description|
|---|---|---|---|---|
|**variable1**|*dtype*|Origin of Data|*Category*|*Description*|
|**variable2**|*dtype*|Origin of Data|*Category*|*Description*|
|**IMAGE_HEIGHT**|*int*|utils.py|*Global Variable*|*160(pixels)-Vertical units across: Top=0 to Bottom= 159*|
|**IMAGE_WIDTH**|*int*|utils.py|*Global Variable*|*320(pixels)-Horizontal units across: Left=0 to Right= 319*|
|**IMAGE_CHANNELS**|*int*|utils.py|*Global Variable*|*3-RGB Channels*|
|**INPUT_SHAPE**|*3-tuple*|utils.py|*Global Variable*|*(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)*|
|**variable2**|*dtype*|Origin of Data|*Category*|*Description*|
|**variable1**|*dtype*|Origin of Data|*Category*|*Description*|


|*VGG-16 Block*|*Name (Type)*|*Kernel Size*|*Nodes*|Params #|*Stride/Pool*|*Output ( h x w x depth )*|
|---|---|---|---|---|---|---|
|**00-First**|**input1 (Input)**|*No Filter*|None|0|None|*( Batch, 224, 224, 3-RGB )*|
|**01-Block 01**|**conv1 (Conv2D)**|*( 3 x 3 )*|64|1,792|*( 1 x 1 )*|*( Batch, 224, 224, 64 )*|
|**02-Block 01**|**conv2 (Conv2D)**|*( 3 x 3 )*|64| 36,928 |*( 1 x 1 )*|*( Batch, 224, 224, 64 )*|
|<span style="color:yellow">**03-Block 01**</span>|<span style="color:yellow">**pool1 (MaxPooling2D)**</span>|<span style="color:yellow">*( 2 x 2 )*</span>|<span style="color:yellow">None</span>|<span style="color:yellow">0</span>|<span style="color:yellow">*( 2 x 2 )*</span>|<span style="color:yellow">*( Batch, 112, 112, 64 )*|
|**04-Block 02**|**conv1 (Conv2D)**|*( 3 x 3 )*|128| 73,856 |*( 1 x 1 )*|*( Batch, 112, 112, 128 )*|
|**05-Block 02**|**conv2 (Conv2D)**|*( 3 x 3 )*|128| 147,584 |*( 1 x 1 )*|*( Batch, 112, 112, 128 )*|
|<span style="color:yellow">**06-Block 02**</span>|<span style="color:yellow">**pool2 (MaxPooling2D)**</span>|<span style="color:yellow">*( 2 x 2 )*</span>|<span style="color:yellow">None</span>|<span style="color:yellow">0</span>|<span style="color:yellow">*( 2 x 2 )*</span>|<span style="color:yellow">*( Batch, 56, 56, 128 )*|
|**07-Block 03**|**conv1 (Conv2D)**|*( 3 x 3 )*|256| 295,168 |*( 1 x 1 )*|*( Batch, 56, 56, 256 )*|
|**08-Block 03**|**conv2 (Conv2D)**|*( 3 x 3 )*|256| 590,080 |*( 1 x 1 )*|*( Batch, 56, 56, 256 )*|
|**09-Block 03**|**conv3 (Conv2D)**|*( 3 x 3 )*|256| 590,080 |*( 1 x 1 )*|*( Batch, 56, 56, 256 )*|
|<span style="color:yellow">**10-Block 03**</span>|<span style="color:yellow">**pool3 (MaxPooling2D)**</span>|<span style="color:yellow">*( 2 x 2 )*</span>|<span style="color:yellow">None</span>|<span style="color:yellow">0</span>|<span style="color:yellow">*( 2 x 2 )*</span>|<span style="color:yellow">*( Batch, 28, 28, 256 )*|
|**11-Block 04**|**conv1 (Conv2D)**|*( 3 x 3 )*|512| 1,180,160 |*( 1 x 1 )*|*( Batch, 28, 28, 512 )*|
|**12-Block 04**|**conv2 (Conv2D)**|*( 3 x 3 )*|512| 2,359,808 |*( 1 x 1 )*|*( Batch, 28, 28, 512 )*|
|**13-Block 04**|**conv3 (Conv2D)**|*( 3 x 3 )*|512| 2,359,808 |*( 1 x 1 )*|*( Batch, 28, 28, 512 )*|
|<span style="color:yellow">**14-Block 04**</span>|<span style="color:yellow">**pool4 (MaxPooling2D)**</span>|<span style="color:yellow">*( 2 x 2 )*</span>|<span style="color:yellow">None</span>|<span style="color:yellow">0</span>|<span style="color:yellow">*( 2 x 2 )*</span>|<span style="color:yellow">*( Batch, 14, 14, 512 )*|
|**15-Block 05**|**conv1 (Conv2D)**|*( 3 x 3 )*|512| 2,359,808 |*( 1 x 1 )*|*( Batch, 14, 14, 512 )*|
|**16-Block 05**|**conv2 (Conv2D)**|*( 3 x 3 )*|512| 2,359,808 |*( 1 x 1 )*|*( Batch, 14, 14, 512 )*|
|**17-Block 05**|**conv3 (Conv2D)**|*( 3 x 3 )*|512| 2,359,808 |*( 1 x 1 )*|*( Batch, 14, 14, 512 )*|
|<span style="color:yellow">**18-Block 05**</span>|<span style="color:yellow">**pool5 (MaxPooling2D)**</span>|<span style="color:yellow">*( 2 x 2 )*</span>|<span style="color:yellow">None</span>|<span style="color:yellow">0</span>|<span style="color:yellow">*( 2 x 2 )*</span>|<span style="color:yellow">( Batch, 7, 7, 512 )|
|**19 4D --> 2D**|**flatten (Flatten)**|*No Filter*|None|0|*None*|*( Batch, 25,088 )*|
|**20-Fully Connected**|**fcon1 (Dense)**|*No Filter*|4,096| 102,764,544 |*None*|*( Batch, 4,096 )*|
|**21-Fully Connected**|**fcon2 (Dense)**|*No Filter*|4,096| 16,781,312 |*None*|*( Batch, 4,096 )*|
|**22-Last Layer**|**Output (Dense)**|*No Filter*|1,000| 4,097,000 |*None*|*( Batch, 1,000 )*|

* NOTE : <br>
    CONV2D: \# Param = [ (Kernel-Size x Channel-Depth)+1 ] x Filters-Nodes<br>
    DENSE : \# Param = [ ( Input Size/Shape ) + 1 ] x Output Size/Shape<br><br>
- Total params: 138,357,544<br>
- Trainable params: 138,357,544<br>
- Non-trainable params: 0<br>


# ![](https://neurohive.io/wp-content/uploads/2018/11/vgg16-neural-network.jpg)

|**CNN Model**|*Split*|*Epoch*|*Loss*|*Accuracy*|
|---|---|---|---|---|
|**Bseline MSE**|*Training*|01|0.0316|0.3251|
|**Bseline MSE**|*Validation*|01|0.0191|0.8220|
|**Bseline MSE**|*Training*|02|0.0266|0.3248|
|**Bseline MSE**|*Validation*|02|0.0205|0.8240|

|**CNN Model**|*Split*|*Epoch*|*Loss*|*Accuracy*|
|---|---|---|---|---|
|**Huber Loss, $\delta$=0.2**|*Training*|01|0.0243|0.3254|
|**Huber Loss, $\delta$=0.2**|*Validation*|01|0.0207|0.8245|
|**Huber Loss, $\delta$=0.2**|*Training*|02|0.0131|0.3247|
|**Huber Loss, $\delta$=0.2**|*Validation*|02|0.0097|0.8235|
|**Huber Loss, $\delta$=0.4**|*Training*|01|0.0158|0.3252|
|**Huber Loss, $\delta$=0.4**|*Validation*|01|0.0093|0.8227|
|**Huber Loss, $\delta$=0.4**|*Training*|02|0.0133|0.3249|
|**Huber Loss, $\delta$=0.4**|*Validation*|02|0.0103|0.8233|
|**Huber Loss, $\delta$=0.6**|*Training*|01|0.0160|0.3252|
|**Huber Loss, $\delta$=0.6**|*Validation*|01|0.0092|0.8225|
|**Huber Loss, $\delta$=0.6**|*Training*|02|0.0135|0.3249|
|**Huber Loss, $\delta$=0.6**|*Validation*|02|0.0103|0.8236|
|**Huber Loss, $\delta$=0.8**|*Training*|01|0.0160|0.3252|
|**Huber Loss, $\delta$=0.8**|*Validation*|01|0.0093|0.8213|
|**Huber Loss, $\delta$=0.8**|*Training*|02|0.0135|0.3249|
|**Huber Loss, $\delta$=0.8**|*Validation*|02|0.0099|0.8236|
|**Huber Loss, $\delta$=1.0**|*Training*|02|0.0134|0.3248|
|**Huber Loss, $\delta$=1.0**|*Validation*|02|0.0097|0.8235|


---
<a id='data_aquisition_and_cleaning'></a>
### Data Aquisition & Cleaning
#### Cloning and Debugging

> * 
> * 
> * 
> * 
    
### Data Aquisition & Cleaning
#### Cloning and Debugging

> * 
> * 
> * 
> * 

#### Cloud Computing / Computing with GPU

> * 
> * 
> * 
> * 

#### Training the CNN

> * Network architecture: X layers, X convolution layers, X fully connected layers
> * and then..
> * 

                        
---
<a id='exploratory_analysis'></a>
### Exploratory Analysis

> * Insert EDA details...
> *
> *

**Data Cleaning and EDA**
- Does the student fix data entry issues?
- Are data appropriately labeled?
- Are data appropriately typed?
- Are datasets combined correctly?
- Are appropriate summary statistics provided?
- Are steps taken during data cleaning and EDA framed appropriately?

### Data Visualization

> * Make some pretty plots with Tableau:

**Visualizations**
- Are the requested visualizations provided?
- Do plots accurately demonstrate valid relationships?
- Are plots labeled properly?
- Plots interpreted appropriately?
- Are plots formatted and scaled appropriately for inclusion in a notebook-based technical report?

---
<a id='findings_and_recommendations'></a>
### Findings and Recommendations

  Answer the problem statement:
> 1. Point 1...
> 2. Point 2...
> 3. Point 3...

---
<a id='next_steps'></a>
### Next Steps:

---
<a id='software_requirements'></a>
### Software Requirements:
https://www.quora.com/What-is-the-VGG-neural-network

---
<a id='acknowledgements_and_contact'></a>
### Acknowledgements and Contact:

External Resources:
* [`High quality images of dogs`] (Unsplash): ([*source*](https://unsplash.com/s/photos/dogs))
* [`VQA labeled images of dogs`] (Visual Genome): ([*source*](https://visualgenome.org/VGViz/explore?query=dogs))
* [`Google Open Source: Dog Detection`] (Open Images): ([*source*](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0bt9lr))
* [`Google Open Source: Dog Segmentation`] (Open Images): ([*source*](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=segmentation&c=%2Fm%2F0bt9lr&r=false))
* [`VGG-19`] (Keras API): ([*source*](https://keras.io/api/applications/vgg/))


Papers:
* `VisualBackProp: efficient visualization of CNNs` (arXiv): ([*source*](https://arxiv.org/pdf/1611.05418.pdf))
* `Very Deep Convolutional Networks For Large-Scale Image Recognition` (arXiv): ([*source*](https://arxiv.org/pdf/1409.1556.pdf))
* `Transfer Learning in Keras with Computer Vision Models` (Machine Learning Mastery): ([*source*](https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/))

    
### Contact:

> * Brandon Griffin ([GitHub](https://github.com/griffinbran) | [LinkedIn](https://www.linkedin.com/in/griffinbran/))

Project Link: ([*source*](https://github.com/griffinbran/machines_best_friend.git))

---
### Submission

**Materials must be submitted by 4:59 PST on Friday, December 10, 2020.**

---