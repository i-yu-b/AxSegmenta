# Machine Learning Engineer Nanodegree
## Capstone Project
Joe Udacity  
April 12th, 2018

## I. Definition

### Project Overview


Neurons are the major cellular component of the central and peripheral nervous systems that transmit and receive signals
to and from the rest of the body. Axons are long, slender projection of neurons, which provide the pathway for signal
 transmission, and therefore allows transmitting information to different neurons, muscles, and glands.
To increase conduction speed along the axons in white matter, myelin surrounds the
axons in a sheath-like structure. Myelin is composed of ~ 80% lipid and 20% protein and forms in a lamellar,
membranous structure and plays the significant role in proper nervous system functioning. [1-3]
Myelin sheath damage or loss of myelin (so-called demyelination) results in diverse symptoms, including
loss of vision/hearing, weakness of arms or legs, cognitive disruption, speech impairment, memory loss,
difficulty coordinating movement or balance disorder.

Also, the amount of myelin (myelin volume fraction - MVF) is the hallmark of many neurodegenerative autoimmune diseases,
including multiple sclerosis, acute disseminated encephalomyelitis, transverse myelitis, Guillain–Barré syndrome,
central pontine myelinosis and many others. [4-5]

All current non-invasive medical imaging techniques can only indirectly investigate myelin and estimate its amount MVF,
and, therefore, they do require some gold standard showing a ground truth MVF to test the precision and sensitivity of
newly developed techniques. One of a gold standards for quantitative MVF validation is histology. Being invasive and
destructive method, it provides direct information on the microscopic structure of cells and tissues. Using electron
microscope one can directly image myelin and estimate its amount in different areas of a nervous system.
To use histology as the gold standard for validation, one needs a robust method to quantitatively analyze electron microscopy
images and to estimate myelin volume fraction. To be more precise, the method should provide a binary mask, which
segments myelin (pixel value = 1) from a non-myelin background (pixel value = 0) and then calculates myelin volume
fraction as a sum of pixels with value equals to 1.

In this project I develop fully-automated robust approach for myelin segmentation using deep learning methods.


### Problem Statement

Across different research groups, there's a variety of independently written in-house code to analyze microscopy images
and to segment myelin from microscopy images. Mostly, images are segmented manually or semi-automatically [6-9].
Manual segmentation provides flexibility for variations in images but is time-consuming and user-dependent.
Conversely, fully automatic segmentation requires no user-input but must be very robust to adapt to variability in
image illumination, structure, etc. Semi-automatic segmentation decreases user-dependency, provides some user-control
to ensure proper segmentation, but still requires a lot of efforts to perform quality control. Recently, deep learning
approaches were successfully used to provide fully-automatically, robust to variability between image illumination, structure
binary masks.


### Metrics
Evaluating the quality of segmentation by choosing an evaluation metric is an important
step in designing a deep learning model. Many evaluation metrics have been used in
evaluating segmentation, [10] and there is no formal way
to choose the most suitable metric(s) for a particular
segmentation task and/or particular data. So most researchers choose the evaluation metrics arbitrarily
or according to their popularity in similar tasks.
The Dice coefficient [11] (DICE), also called the overlap index, is the most used metric in validating biomedical
segmentations. DICE measures the spatial overlap between two masks, X and Y target regions, and is defined as
![equation](http://latex.codecogs.com/gif.latex?%5Cfrac%7B2%5Cleft%20%7C%20X%20%5Cright%20%7C%20%5Ccap%20%5Cleft%20%7C%20Y%20%5Cright%20%7C%7D%7B%5Cleft%20%7C%20X%20%5Cright%20%7C&plus;%5Cleft%20%7C%20Y%20%5Cright%20%7C%7D),
where ∩ is the intersection.


## II. Analysis

### Data Exploration

In the process of REMMI project development at Vanderbilt University under supervision of Prof. Mark Does my colleagues
collected electron microscopy images for 6 control rats in 4 different brain regions (the genu, mid-body, and splenium
of the corpus callosum and the anterior commissure). Ultra-thin sections of brains (~ 500 x 500 x 0.07 μm) were imaged
on the Philips/FEI Tecnai T12 electron microscope (FEI Company, Hillsboro, OR) at 15,000x magnification and pictures
were acquired with a side-mounted AMT CCD camera. For quantification 6-12 high-resolution images were collected (~300 axons)
per ROI per animal. So, in total there are 141 high resolution 2048 × 2048 raw images in greyscale. Each image was analyzed
semi-automatically to derive a ground-truth binary mask and to estimate myelin volume fraction.

Therefore, to create deep learning neural network for electron microscopy images segmentation
I had relatively big training dataset, consisting of high-resolution images of white matter and corresponding manually
corrected binary masks for myelin segmentation.


### Exploratory Visualization
As it was said in previous section, there are 141 high resolution 2048 × 2048 raw images and corresponding binary masks.
Typically, there is a gray background with different dark-gray blotches (nucleii, proteins, etc) and almost black
semi-circular shaped myelin layers. As you can see from example images, myelin layers vary in size, shape, thickness,
and often overlaps with each other, making classical computer vision algorithms (watershed, thresholding) fail.

![Examples of high-res images and corresponding masks](image1_full_image.png?raw=true "Examples of high-res images and corresponding masks")

To train
deep neural nets we definitely need a smaller size images to fit them into memory. Therefore, from each high-resolution
image I randomly cropped 100 patches of size 224*224. So in total there are 5962 images with corresponding binary masks.
In average, there are around 1-3 fragments of different axons on each image.

![Examples of patched images and corresponding masks](image1_patches_image2.png?raw=true "Examples of patched images and corresponding masks")

The whole dataset then was randomly splitted
into train, validation and test datasets in 70:20:10 ration, having 4173 images in train dataset, 1192 images in validation dataset,
597 images in test dataset.


### Algorithms and Techniques

The problem of building binary masks of images is actually a sub-class of semantic segmentation tasks.
Semantic segmentation is understanding an image at pixel level i.e, we want to assign each pixel in the image to an object class.
As with image classification, convolutional neural networks (CNN) have had enormous success on segmentation problems.
One of the most successful for semantic segmentation challenge CNN approaches is encoder-decoder architecture.
Encoder gradually reduces the spatial dimension by using pooling layers and decoder gradually recovers the object details and
spatial dimension. To restore spatial information normally shortcut connections from encoder to decoder  are used.
The most popular encoder-decoder architecture for biomedical challenges is U-Net architecture [12].
As U-net up to now is the best method on the ISBI challenge for segmentation of neuronal structures in electron microscopic stacks; it has won the Grand Challenge for Computer-Automated Detection of Caries in Bitewing Radiography at ISBI 2015, and it has won the Cell Tracking Challenge at ISBI 2015 on the two most challenging transmitted light microscopy categories (Phase contrast and DIC microscopy) by a large margin; has been actively used in Kaggle competitions for
image segmentation tasks. [13]
As U-net was originally designed for electron microscopy segmentation tasks, I've decided to implement that solution for my goals.
<p align="center">
    <img align="center" src="https://github.com/i-yu-b/machine-learning/blob/master/projects/capstone/Screen%20Shot%202017-11-30%20at%205.08.12%20PM.png" width="500" hspace="20"/>
</p>

I have used the original architecture with some modification: 6 convolutional layers in both encoder and decoder, with
32, 64, 128, 256, 512 and 1024 filter. The detailed architecture with all the parameters is the following:
<p align="center">
    <img align="center" src="https://github.com/i-yu-b/AxSegmenta/blob/master/Report/model.png" width="500" hspace="10"/>
</p>

### Benchmark
As a benchmark model, I will use an approach which is often employed as a good start for manual and
semi-automatic methods - global thresholding. []
The general idea is to apply a threshold to each image based on the nadir of the histogram between the two peaks of myelin and non-myelin pixels. Since the myelin is stained in histology images, it is dark in the image and captured in the first signal peak.
In the binary image, all pixels falling below the threshold = 1 and are considered myelin, while
all pixels above the threshold = 0 and are considered non-myelin. While
this technique was a good start, it does not work consistently, produces not smooth, sharp masks


In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?

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
10. [Taha AA, Hanbury A. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC Medical Imaging. 2015](https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-015-0068-x)
11. Dice LR. Measures of the amount of ecologic association between species. Ecology. 1945;26(3):297–302. doi: 10.2307/19324094
12. [Olaf Ronneberger, Philipp Fischer, Thomas Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
13. [University of Freiburg. U-Net: Convolutional Networks for Biomedical Image Segmentation](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

