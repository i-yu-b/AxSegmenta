# Machine Learning Engineer Nanodegree
## Capstone Proposal
Irina Barskaya
April 5th, 2018

## Proposal

### Domain Background

Neurons are the major cellular component of the central and peripheral nervous systems that transmit and receive signals to and from the rest of the body.
Axons are long, slender projection of neurons, which provide the pathway for signal transmission, and therefore allows transmitting information to different neurons, muscles, and glands.
To increase conduction speed along the axons in white matter, myelin surrounds the
axons in a sheath-like structure. Myelin is composed of ~ 80% lipid and 20% protein and forms in a lamellar,
membranous structure and plays the significant role in proper nervous system functioning. [1-3]
Myelin sheath damage or loss of myelin (so-called demyelination) results in diverse symptoms, including
loss of vision/hearing, weakness of arms or legs,
cognitive disruption, speech impairment, memory loss, difficulty coordinating movement or balance disorder.

Also, the amount of myelin (myelin volume fraction - MVF) is the hallmark of many neurodegenerative autoimmune diseases, including multiple sclerosis, acute disseminated encephalomyelitis, transverse myelitis, Guillain–Barré syndrome,
 central pontine myelinosis and many others. [4-5]
 One of the main methods for non-invasive MVF estimations is quantitative MRI.
Because myelin is mainly a lipid structure, which has a very short-lived MRI
signal from lipids, the direct MRI imaging of myelin is extremely difficult.

Most of current
techniques can only indirectly investigate myelin and estimate its amount MVF and they require some gold standard
showing a ground truth MVF to
test the precision and sensitivity of newly developed techniques.
One of a gold standards for quantitative MRI validation is histology. Being invasive and destructive method
it provides direct information on the microscopic structure of cells and tissues. Using electron microscope one can directly image myelin and estimate its amount in different areas of a nervous system.

### Problem Statement

To use histology as the gold standard for validation of MRI parameters, one
needs a robust method to quantitatively analyze electron microscopy
images and to estimate myelin volume fraction. To be more precise, the method
should provide a binary mask, which segments myelin (pixel value = 1) from a non-myelin background
(pixel value = 0) and then calculates myelin volume fraction as a sum of pixels with value equals
to 1.

Across different research groups, there's a variety of independently written in-house code to analyze microscopy images.
Mostly, images are segmented manually or semi-automatically [6-9].
Manual segmentation provides flexibility for variations in images but is time-consuming and
user-dependent. Conversely, fully automatic segmentation requires no user-input but must be
very robust to adapt to variability in image illumination, structure, etc. Semi-automatic
segmentation decreases user-dependency, provides some user-control to ensure
proper segmentation, but still requires a lot of efforts to perform quality control.
Recently, deep learning approaches were successfully used to provide
fully-automatically, robust to variability between image illumination, structure
binary masks.

### Datasets and Inputs

In the process of REMMI project development at Vanderbilt University under supervision of Prof. Mark Does my colleagues collected electron
microscopy images for 6 control rats in 4 different brain regions (the genu, mid-body, and splenium of the corpus callosum and the anterior commissure).
Ultra-thin sections of brains (~ 500 x 500 x 0.07 μm) were imaged on the Philips/FEI Tecnai T12 electron microscope (FEI Company, Hillsboro, OR) at 15,000x magnification and pictures were acquired with a side-mounted AMT CCD camera. For quantification 6-12 high-resolution images were collected (~300 axons) per ROI per animal. The  raw images are greyscale and have 2048 × 2048 resolution. After that, each image was analyzed semi-automatically to derive a binary mask and to estimate myelin volume fraction.
Therefore, to create deep learning neural network for electron microscopy images segmentation
I had relatively big training dataset, consisting of high-resolution images of white matter and corresponding manually corrected binary masks for myelin segmentation.

<img src="https://github.com/i-yu-b/machine-learning/blob/master/projects/capstone/control_raw_image_0.png" width="300" hspace="20"/> <img src="https://github.com/i-yu-b/machine-learning/blob/master/projects/capstone/control_raw_mask_0.png" width="300"/>

### Solution Statement

The problem of building binary masks of images is actually a sub-class of semantic segmentation tasks.
Semantic segmentation is understanding an image at pixel level i.e, we want to assign each pixel in the image to an object class.
As with image classification, convolutional neural networks (CNN) have had enormous success on segmentation problems.
Currently, two main CNN approaches are most successfully used for semantic segmentation challenge.

First one is encoder-decoder architecture. Encoder gradually reduces the spatial dimension by using pooling layers and decoder gradually recovers the object details and spatial dimension. To restore spatial information normally shortcut connections from encoder to decoder  are used. The most popular encoder-decoder architecture for biomedical challenges is U-Net architecture [10].
As U-net up to now is the best method on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks; it has won the Grand Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted light microscopy categories (Phase contrast and DIC microscopy) by a large margin; has been actively used in Kaggle competitions for
image segmentation tasks. [11]

Architectures of the second class use dilated/atrous convolutions. Dilated convolutional layer allows an exponential increase in field of view without decrease of spatial dimensions. [12]

As U-net was originally designed for electron microscopy segmentation tasks, I've decided to implement that solution for my goals.

<img align="center" src="https://github.com/i-yu-b/machine-learning/blob/master/projects/capstone/Screen%20Shot%202017-11-30%20at%205.08.12%20PM.png" width="500" hspace="20"/>

### Benchmark Model

As a benchmark model, I will use an approach which is often employed as a good start for manual and
semi-automatic methods - global thresholding.
The general idea is to apply a threshold to each image based on the nadir of the histogram between the two peaks of myelin and non-myelin pixels. Since the myelin is stained in histology images, it is dark in the image and captured in the first signal peak.
In the binary image, all pixels falling below the threshold = 1 and are considered myelin, while
all pixels above the threshold = 0 and are considered non-myelin. While
this technique was a good start, it does not work consistently, produces not smooth, sharp masks with artefacts due to dark background features, like nuclei, and always requires manual correction.

### Evaluation Metrics

Evaluating the quality of segmentation by choosing an evaluation metric is an important
step in designing a deep learning model. Many evaluation metrics have been used in
evaluating segmentation, [13] and there is no formal way
to choose the most suitable metric(s) for a particular
segmentation task and/or particular data. So most researchers choose the evaluation metrics arbitrarily
or according to their popularity in similar tasks.
The Dice coefficient [14] (DICE), also called the overlap index, is the most used metric in validating biomedical segmentations. DICE measures the spatial overlap between two masks, X and Y target regions, and is defined as ![equation](http://latex.codecogs.com/gif.latex?%5Cfrac%7B2%5Cleft%20%7C%20X%20%5Cright%20%7C%20%5Ccap%20%5Cleft%20%7C%20Y%20%5Cright%20%7C%7D%7B%5Cleft%20%7C%20X%20%5Cright%20%7C&plus;%5Cleft%20%7C%20Y%20%5Cright%20%7C%7D),  where ∩ is the intersection.


### Project Design
In order to perform accurate and robust binary segmentation of electron microscopy I will use convolutional neural network with U-net architecture. The first stage of the project is data preprocessing: from each high resolution image (2048 × 2048) I randomly crop 100 patches with resolution 224 x 224. Most likely to increase the robustness and the accuracy of the model additional data augmentation, such as shift, zoomed in/out, rotation, flip, affine transformation, elastic transformation will be required. After preprocessing the data it will be splitted in 3 datasets (train, validation, test) with 70/20/10 ratio.
Then I will implement basic U-net model in Keras+Tensorflow and launch the learning process.
In the training runs I will vary the following hyperparameters: batch normalization, dropout,
learning rate.


### References
1. Morell P, Quarles R, Norton W. Formation, structure, and biochemistry of myelin. In: 4th ed.
New York: Raven Press Ltd; 1989. pp. 109–136.
2. Trapp BD, Kidd G. Structure of the myelinated axon. In: London: Elsevier Academic Press; 2004. pp. 3–27.
3. Van De Graff K. Nervous tissue and the central nervous system. In: New York: McGraw-Hill; 2002. p. 351.
4. Trapp BD, Ransohoff R, Rudick R. Axonal pathology in multiple sclerosis: relationship to
neurologic disability. Current opinion in neurology 1999;12:295–302.
5. Simao G, Raybaud C, Chuang S, Go C, Snead O, Widjaja E. Diffusion Tensor Imaging of
Commissural and Projection White Matter in Tuberous Sclerosis Complex and
Correlation with Tuber Load. American Journal of Neuroradiology 2010;31:1273–77.
6. Jelescu I, Zurek M, Winters K, et al. In vivo quantification of demyelination and recovery
using compartment-specific diffusion MRI metrics validated by electron microscopy.
Neuroimage 2016;132:104–14.
7. Stikov N, Campbell JS, Stroh T, Lavelée M, Frey S, Novek J, Nuara S, Ho M-K, Bedell BJ,
Dougherty RF. In vivo histology of the myelin g-ratio with magnetic resonance imaging.
NeuroImage 2015;118:397–405.
8. Dula AN, Gochberg DF, Valentine HL, Valentine WM, Does MD. Multiexponential T2,
magnetization transfer, and quantitative histology in white matter tracts of rat spinal cord.
Magnetic Resonance in Medicine 2010;63:902–9. doi: 10.1002/mrm.22267.
9. West K, Kelm N, Carson R, Does M. A revised model for estimating g-ratio from MRI.
NeuroImage 2016;125:1155–8.
10. [Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
11. [University of Freiburg. U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
12. [Fisher Yu, Vladlen Koltun. Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122)
13. [Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC Medical Imaging. 2015](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-015-0068-x)
14. Dice LR. Measures of the amount of ecologic association between species. Ecology. 1945;26(3):297–302. doi: 10.2307/19324094
