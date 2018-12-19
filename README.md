# cosmo_project
NMR spectroscopy is a well-known and highly efficient method for probing the magnetic fields between atoms and determining how neighboring atoms interact with each other. However, full crystal structure determination by NMR spectroscopy requires extremely complicated, time-consuming calculations involving quantum chemistry - nearly impossible for molecules with very intricate structures". However the use of Machine Learning appears significant for this task and provides good results. Indeed physics phenomenons always follow rules that can be found in the underlying data. The COSMO lab already achieved an RMSE of 0.6. They challenged us to find more efficient regression algorithms, or to try what neural networks could bring to the problem. Following the _No Free Lunch Theorem_ , we wanted to try as much methods as possible in order to find the best one for the given problem.

This README file details the different implementations we went through to solve this problem as well as the files in which they are located.

## Data Processing and Features Reduction

The code used to generate our reduced matrix can be found in the notebook PreProcessing.ipynb. It is shown there that the main algorithm used is a Principal Component Analysis, which allows us to reduce the number of features from 15k to 4k and still keep the same Root-Mean-Squarre-Error when appliying Rigde Regression with a very small lambda. Normalization and whitening was also applyied to the data to ensure the best predictions possible.

## Machine Learning

## Deep Learning

The application of the code for our work regarding Deep Learning is located in the __Deep_Learning_Approch.ipynb__ notebook. There you will find all our tests, from the __single neural network__ to the __supercrowd__ along with the different optimizations we made.

However, there's some implementations that needed a bit more application and modularity, namely the implementation of the __Crowd__, the __Experts__, the __CollaborativeCrowd__ and finally the __SuperCrowd__.

### Crowd predictions

Based on the principle of the _Wisdom of Crowds_, we created a crowd of neural networks which will reduce the variance of our predictions and hence reduce the overall error. This specific implementation necessited a class to itself for modularity and cleaness reasons. All the code for this class is located in the file crowd.py. Note that the created crowds are automatically saved in the _session_ folder to avoid redoing heavy computations. It has also been made so that it is highly parametrable, again to enforce modularity.

### Experts predictions

As suggested by Nowlan and Hinton in _Evaluation of Adaptive Mixtures of Competing Experts_, we tried to specialize our neural networks to perform well on a given subset of the data. This technique assumes that the data is categorizable. With this idea in mind, we wrote the class _Experts_ located in experts.py. It is similar to the Crowd but there's a twist : instead of making a simple average of the predictions, we select the neural networks that should perform best on the given data for each prediction. 

The categorization has been made in two different fashion :
- Using a deep learning classifier similar to the ones used for image classification and compute a vector of probabilities assigning to each neural network the probability that it will perform well on the new sample (method __predict__).
- Using a k-means algorithm to cluster the data prior to the prediction, then evaluate which neural networks perform best on each cluster and finally at prediction time, assign the sample to evaluate to closer cluster and use the related neural networks to obtain the prediction.

However, none of this techniques gave us good results, or at least results that are not better than what we had before. This may be explained by the fact that our data is not easily categorizable which was a prerequisite for the method to work efficiently. Hence we dropped the idea of improving our predictions using this technique but we let it in this notebook for completness.

### Collaborative Crowd

In this approach, we try to make each neural network joining the crowd learn from the predictions of the previously added networks, hoping that it will reduce our error even further. Hence the predictions of the others are added to the training and prediction matrix. The code for this class is located in collaborative_crowd.py.

This approach gives us results that are a bit better than the ones of the simple crowd, however there's no break through.

### Super Crowd

The idea behind this one is rather simple : we take several highly precise crowds and combine them into one through a Composite Design Pattern. The prediction is the average of the predictions of the crowds composing the supercrowd. 
The super crowd has the following property : if rmse(c1) < a, rmse(c2) < a for a constant a and two crowds c1, c2, then rmse(supercrowd(c1, c2)) < a. Hence our error can only decrease.

Note also that a super crowd may be composed of supercrowds.

This approach decreased our error again by a small amount, still noticeable. However note that the computation time is the sum of the computation time of the crowds composing the super crowd.
