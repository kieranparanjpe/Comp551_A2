from pathlib import Path

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

class Dataset:

    def __init__(self):
        rng = np.random.default_rng(2026)

        # Resolve paths relative to project root (parent of src/)
        project_root = Path(__file__).resolve().parent.parent
        data_file = project_root / "spambase" / "spambase.data"
        names_file = project_root / "spambase" / "spambase.names"

        if data_file.is_file() and names_file.is_file():
            self.X, self.y = self._load_spambase_local(data_file, names_file)
        else:
            # fetch dataset online
            spambase = fetch_ucirepo(id=94)

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

    def _load_spambase_local(self, data_path: Path, names_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Parse .names for attribute names (ignore comments and non-attribute lines).
        attr_names = []
        with open(names_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("|"):
                    continue
                if ":" in line:
                    name = line.split(":", 1)[0].strip()
                    if name:
                        attr_names.append(name)

        df = pd.read_csv(data_path, header=None)
        if df.shape[1] < 2:
            raise ValueError(f"Local spambase file has too few columns: {df.shape}")

        # If .names includes the class, use it. Otherwise, append a class name.
        if len(attr_names) == df.shape[1] - 1:
            attr_names.append("class")
        elif len(attr_names) != df.shape[1]:
            # Fallback to generic names if parsing was off.
            attr_names = [f"f{i}" for i in range(df.shape[1] - 1)] + ["class"]

        df.columns = attr_names
        X = df.iloc[:, :-1].copy()
        y = df.iloc[:, -1].to_frame(name="class")
        return X, y
