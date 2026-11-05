# SMILES Toxicity Predicton

App Link: https://brandonwalk-smiles-toxicity-appapp-vmdjxi.streamlit.app

Anisha Jaiswal  
Pei Yu Hsueh
Srish Maulik
Brandon Walker

## Decision Threshold

The decision threshold was set to 0.45. This was done for two reasons:
1. We decided to go after recall as a metric to optimize, hoping that our team would have this as a unique target. When the threshold is lowered the recall gets higher at the cost of specificity. This would be fine in the real world as more things are identified as toxic and thus need to be tested further.
2. The specific value of 0.45 was done after re-training the model multiple times under different random seeds for the train test split until we could ensure that 10/10 random seed starts would give us a specificity >0.7.

## Feature Engineering

We had tried two methods for generating numeric data.
1. A doc2vec model (from the gensim package) built on the SMILES strings. We incorporated more SMILES strings from outside data to train the doc2vec featurize model. This predicts tokens around other tokens. It did work, but the random forest trained on top of the vectors barely gave a AUCROC above 0.85, so we deemed it not good enough.
2. Morgan Fingerprints was something I had read about in the book/package [DeepChem](https://deepchem.io/tutorials/the-basic-tools-of-the-deep-life-sciences/) which explains how the atoms and the neighbors of atoms are hashed to create somewhat unique features (there can be hash collisions) to train on. The optimization of the FP size was attempted. Because we ultimately used a tree based model that uses the square root method to consider features, having a large FP size is not necessary because the model isn't considering a lot of the finger print. We thus tried to optimize the FP size and found by reducing it to 1024 there was improvement in speed of the model, no meaningful drop in accuracy. Dropping the FP size further didn't do anything, but I suspect it would if we had a larger dataset (and thus more potential hash collisions).

## Model Selection and Hyperparameters

A tree based model is a good model to move forward with once you have done fingerprints. Interaction effects between any two variables in a fingerprint is basically impossible to intuit because there is nothing meaningful about the fingerprint. With the fingerprint essentially being binary a model like logisitic regression is also not useful. We did optimize the number of trees in the random forest, diminshing them did not have a large impact from our starting point of 300 trees. The speed can get quite fast with <50 trees but metrics did suffer at that point.

## AI Usage

Early version of the doc2vec and streamlit app were AI free, but other team members did use AI for the creation of their model. Integration of someone else's model was difficult so Claude Code was used to ensure that the model worked as intended in the streamlit app. Optimization of the hyperpatemers (trees, threshold, FP size) was done without AI and just repeated many times locally with different random seeds for the train test split. I also prompted Claude for ways to cache the featurizer and/or model within the app to make it more responsive, whether or not that is where speed is tested from it does improve the perofrmance of the app. While the instructions does say to not have many non-essential elements I did ask Claude to add the visuals at the bottom in case this does get put on my resume, I'd like it to look nice.