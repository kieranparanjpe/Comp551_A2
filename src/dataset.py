import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd

class Dataset:

    def __init__(self):
        # fetch dataset
        spambase = fetch_ucirepo(id=94)
        rng = np.random.default_rng(2026)

        # data (as pandas dataframes)
        self.X: pd.DataFrame = spambase.data.features
        self.y: pd.DataFrame = spambase.data.targets
        perm = rng.permutation(len(self.X))
        self.X = self.X.iloc[perm].reset_index(drop=True)
        self.y = self.y.iloc[perm].reset_index(drop=True)

        self.trainX = np.zeros(0)
        self.trainY = np.zeros(0)
        self.testX = np.zeros(0)
        self.testY = np.zeros(0)

        self.preprocess()

        self.numFeatures = len(self.X.columns)

    def preprocess(self):
        self.X.insert(0, "bias", 1)

        self.trainX = self.X[0:230].to_numpy()
        self.trainY = self.y[0:230].to_numpy()
        self.testX = self.X[230:].to_numpy()
        self.testY = self.y[230:].to_numpy()

        # Standardize features using training set stats. Keep bias column unchanged.
        train_feats = self.trainX[:, 1:]
        mean = train_feats.mean(axis=0)
        std = train_feats.std(axis=0)
        std[std == 0] = 1.0

        self.trainX[:, 1:] = (train_feats - mean) / std
        self.testX[:, 1:] = (self.testX[:, 1:] - mean) / std

