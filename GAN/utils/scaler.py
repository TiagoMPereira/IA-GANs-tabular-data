import numpy as np
import pandas as pd


class Scaler:
    def init(self):
        self.scaled_dict = {}

    def fit(self, data):
        self.scaled_dict = {
            col: {
                "mean": np.mean(data[col]),
                "var": np.var(data[col])
            } for col in data.columns
        }

    def transform(self, data):
        transformed_data = data.copy()
        for col in data.columns:
            values = data[col]
            transformed = (values - self.scaled_dict[col]["mean"]) / self.scaled_dict[col]["var"]
            transformed_data[col] = transformed

        return transformed_data
    
    def inverse_transform(self, transformed_data):
        data = transformed_data.copy()
        for col in transformed_data.columns:
            values = transformed_data[col]
            reverse = values * self.scaled_dict[col]["var"] + self.scaled_dict[col]["mean"]
            data[col] = reverse

        return data