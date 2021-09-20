# GAN_RoBERTa

### Introduction

Code repository for the paper: <br/>
<b>GAN-RoBERTa: a Robust Semi-Supervised Model for Detecting Anti-Asian COVID-19 Hate Speech on Social Media</b>

GAN-RoBERTa is an effective semi-supervised generative adversarial network based methodology for COVID-19 related hate speech detection. GAN-RoBERTa utilizes the RoBERTa encoder as a feature extractor to generate latent embedding from labelled([EA dataset](https://zenodo.org/record/3816667#.YUhhjrgzaUk)) and unlabelled data([CLAWS dataset](http://claws.cc.gatech.edu/covid/#dataset)) in the training process. Our experiment results indicate that with the specific participation of unlabelled data in the optimization process, the model's Macro-F1 score is significantly boosted from 0.77 to 0.82.
<p align="center">
  <img src="Figure/GAN_Roberta.png" width=700 align="center">
</p>

#### GAN-RoBERTa consists of 3 main components: 
(1) A RoBERTa encoder extracts sentence-level features from labelled B<sub>1</sub> * S and unlabelled data B<sub>2</sub> * S, along with an extra linear layers stack that compress the sentence-level features into B * H<sub>2</sub>.  <br/>
(2) A generator that transforms the random noise B * H<sub>1</sub> into B * H<sub>2</sub>.  <br/>
(3) A discriminator classifies the sentence-level embedding into three categories (Hate, Non-hate, Fake). Each linear layer is concatenated with the Leaky-ReLU activation layer and dropout.

In the following plots, the performances of GAN-RoBERTa are reported for [2400 public annotated dataset](http://claws.cc.gatech.edu/covid/#dataset) from [Racism is a Virus: Anti-Asian Hate and Counterhate in Social Media during the COVID-19 Crisis](https://arxiv.org/abs/2005.12423). We compared in detail the effects of models with different ratio of unlabeled data on the 2400 test set.



In this repository we provide our research code for training and testing.
