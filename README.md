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
- Transfer Learning with a pre-trained deep learning image classifier (VGG-16 CNN from Visual Geometry Group in 2016)

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
> * 
> * 
> * 
> * 
> * 

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


|**CNN Architecture**|*Kernel Size*|*neurons*|No. of Images|*Stride*|*Shape ( h x w x RGB )*|
|---|---|---|---|---|---|
|**Input Layer**|*None*|None|< sample size >|None|*( 160 x 320 x 3 )*|
|**Convolution 01**|*( 5 x 5 )*|24|24|*( 2 x 2 )*|*(  78 x 158 x 3 )*|
|**Convolution 02**|*( 5 x 5 )*|36|864|*( 2 x 2 )*|*(  37 x  77 x 3 )*|
|**Convolution 03**|*( 5 x 5 )*|48|41,472|*( 2 x 2 )*|*(  16 x  36 x 3 )*|
|**Convolution 04**|*( 3 x 3 )*|64|2,654,208|*None*|*(  37 x  77 x 3 )*|
|**Convolution 05**|*( 3 x 3 )*|64|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Dropout**|*None*|None|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Flatten**|*None*|None|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Dense 01**|*None*|100|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Dense 02**|*None*|50|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Dense 03**|*None*|10|169,869,312|*None*|*(  16 x  36 x 3 )*|
|**Dense Output**|*None*|1|169,869,312|*None*|*(  16 x  36 x 3 )*|


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

### Contact:

> * Brandon Griffin ([GitHub](https://github.com/griffinbran) | [LinkedIn](https://www.linkedin.com/in/griffinbran/))

Project Link: ([*source*](https://github.com/griffinbran/machines_best_friend.git))

---
### Submission

**Materials must be submitted by 4:59 PST on Friday, December 10, 2020.**

---