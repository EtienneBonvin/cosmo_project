# NMR spectroscopy using Machine Learning
This project has been realized in the context of the graduate Machine Learning course given at EPFL (CS-433). In collaboration with the [Laboratory of Computational Science and Modelling (COSMO)](https://cosmo.epfl.ch/), our goal was to predict electromagnetic shielding using some dataset provided by the lab itself.

We shaped the project around three main axes:

- Data analysis and feature reduction (see [PreProcessing.ipynb](https://github.com/EtienneBonvin/cosmo_project/blob/master/PreProcessing.ipynb))

    The dataset initially contains more than 30k samples and 15k features. In order to make our computations faster and to reduce the risk of overfitting, we first tried to lower the number of features without affecting negatively the quality of our predictions.

- Standard machine learning approaches (see [ML.ipynb](https://github.com/EtienneBonvin/cosmo_project/blob/master/ML.ipynb))

    Our main goal for this part was to validate the results that the lab had obtained: RMSE of 0.6 using kernel ridge regression. In order to so, we tried different machine learning algorithms such as `Least Squares`, `Ridge Regression` and `SGD` using different loss functions.

- Deep learning (see [DL.ipynb](https://github.com/EtienneBonvin/cosmo_project/blob/master/DL.ipynb))

    In parallel of standard machine learning, we decided to run a few deep learning experiments, which the lab had not tried yet, to see whether we could reach better results.

Implementation and results for each part are explained in the different subsections below.

The file [run.py](https://github.com/EtienneBonvin/cosmo_project/blob/master/run.py) is meant to produce our best results and our report in PDF format can be found [here](https://github.com/EtienneBonvin/cosmo_project/blob/master/report.pdf). The dataset, however, is still being processed by the lab and is not yet ready to be made publicly available.

## `run.py` in a nutshell
We have declined `run.py` in two flavors. Since we have tried both machine learning and deep learning approaches, we wanted this script to be able to reproduce our best results for both.

Launching it can be done as follows:
```
python run.py [DL|ML]
```
If no option is provided, it will produce results for deep learning by default. If you want to see our results for the machine learning side of the project, you can run it using the appropriate option.

## 1. Data Processing and Features Reduction
The code used to generate our reduced data matrix can be found in [PreProcessing.ipynb](https://github.com/EtienneBonvin/cosmo_project/blob/master/PreProcessing.ipynb). It is shown there that the main algorithm used is `Principal Component Analysis` (PCA), which allows us to reduce the number of features from 15k to 4k while keeping the same RMSE as `Ridge Regression` with a very small regularizer lambda. `Normalization`, `whitening` and `jittering` were also applied to the data to ensure the best predictions possible, but did not always reveal to be much useful. Finally, it turned out that the matrix producing the best results was not the same for machine learning than for deep learning. We explain below how to generate the matrix for both methods:

### 1.1 Machine learning dataset
- PCA reducing the number of components to 4500 features
- Normalization
- Jittering (1% of new samples created)
### 1.2 Deep learning dataset
- PCA reducing the number of components to 3004 features
- Normalization

The jittering was not giving optimal results for deep learning, we suppose that this may be linked to the fact that our trained model notices the small noise that we added and has a harder time finding the relations mappping inputs to outputs.
3000 features were enough to reproduce the RMSE achieved with the full matrix for deep learning, but we noticed significant improvement on the Machine Learning side when keeping 4500 features instead of 3000.

## 2. Machine Learning
The machine learning part ([ML.ipynb](https://github.com/EtienneBonvin/cosmo_project/blob/master/ML.ipynb)) was intended to reproduce results obtained from the lab in the first place and to improve them in a second time. In order to do so, we tried `Least Squares` to get a first hand-on the data, before we decided to move to `Ridge Regression` with `Polynomial Expansion`. Using the appropriate data matrix, this gave us our best result: __RMSE of 0.53__. For completeness, we also implemented the `Lasso` and `MAE loss` but those two methods were not really conclusive. In the first case, we saw that our SGD was either converging too slowly or was too hard to tune (especially the learning factor). In the second case, it simply turns out that `MAE` is not a loss function that suits our problem well. Concrete implementations of the algorithms can be found in [regressions.py](https://github.com/EtienneBonvin/cosmo_project/blob/master/src/regressions.py).


## 3. Deep Learning
[DL.ipynb](https://github.com/EtienneBonvin/cosmo_project/blob/master/DL.ipynb) shows the code for our work regarding deep learning. In particular, we tried structures starting from `Single Neural Networks` to  `Supercrowds` along with the different optimization methods.

Some implementations such as `Crowd`, `Experts`, `CollaborativeCrowd` and `SuperCrowd` needed a bit more work and modularity. The way they work is decribed in the following subsections.

### 3.1 Crowd predictions
Based on the principle of the _Wisdom of Crowds_, we created a crowd of neural networks which will reduce the variance of our predictions and hence reduce the overall error. This specific implementation required a class to itself for modularity and cleanness reasons. All the code for this class is located in the file [crowd.py](https://github.com/EtienneBonvin/cosmo_project/blob/master/src/crowd.py). Note that the created crowds are automatically saved in the `session/` folder to avoid redoing heavy computations. It has also been made so that it is highly parametrable, again to increase modularity.

### 3.2 Experts predictions
As suggested by Nowlan and Hinton in _Evaluation of Adaptive Mixtures of Competing Experts_, we tried to specialize our neural networks to perform well on a given subset of the data. This technique assumes that the data is categorizable. With this idea in mind, we wrote the class _Experts_ located in [experts.py](https://github.com/EtienneBonvin/cosmo_project/blob/master/src/experts.py). It is similar to the crowd but there's a twist : instead of making a simple average of the predictions, we select the neural networks that should perform best on the given data for each prediction. 

The categorization has been made in two different fashions :
- Using a deep learning classifier similar to the ones used for image classification and compute a vector of probabilities assigning to each neural network the probability that it will perform well on the new sample (method `predict`).
- Using a k-means algorithm to cluster the data prior to the prediction, then evaluate which neural networks perform best on each cluster and finally, at prediction time, assign the sample to evaluate to closer cluster and use the related neural networks to obtain the prediction.

However, none of this techniques gave us good results, or at least results that are not better than what we had before. This may be explained by the fact that our data is not easily categorizable which was a prerequisite for the method to work efficiently. Hence we dropped the idea of improving our predictions using this technique but we keep it in this repository for completeness.

### 3.3 Collaborative Crowd
In this approach, we try to make each neural network joining the crowd learn from the predictions of the previously added networks, hoping that it will reduce our error even further. Hence the predictions of the others are added to the training and prediction matrix. The code for this class is located in [collaborative_crowd.py](https://github.com/EtienneBonvin/cosmo_project/blob/master/src/collaborative_crowd.py).

This approach gives us results that are a bit better than the ones of the simple crowd, however there's no break through.

### 3.4 SuperCrowd
The idea behind this one is rather simple : we take several highly precise crowds and combine them into a single one through a Composite Design Pattern. The prediction is the average of the predictions of the crowds composing the supercrowd. 
The super crowd has the following property : if RMSE(c1) < a, RMSE(c2) < a for a constant a and two crowds c1, c2, then RMSE(supercrowd(c1, c2)) < a. Hence our error can only decrease.

Also note that a super crowd may be composed of supercrowds. the code for this class is located in [supercrowd.py](https://github.com/EtienneBonvin/cosmo_project/blob/master/src/supercrowd.py).

This approach decreased our error again by a small amount, still noticeable. However the computation time is the sum of the computation time of the crowds composing the super crowd.

## Conclusion
Among all methods that we have tried and cross-validated, the one that worked best is the __SuperCrowd__ composed of a __CollaborativeCrowd__ and a __simple crowd__ producing a __RMSE of 0.342__, which is almost the half of what first goal was. Moreover, it's important to be noted that this results has been obtained without any domain-specific knowledge. Therefore, we expect the COSMO to be able to increase the precision of the predictions by combining our algorithms with their understanding of the involved scientific material. Finally, we have tried different feature reduction techniques and showed which worked well and which not, which is also a valuable information because of evident performance reasons.

> "Go TF..."
Michele Ceriotti
