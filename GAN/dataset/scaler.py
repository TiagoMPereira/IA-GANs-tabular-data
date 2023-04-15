import pandas as pd
import numpy as np
import warnings


class Column():

    def __init__(self, name: str):
        self.name = name
        self.type = None

    def fit(self):
        pass
    
    def transform(self):
        pass

    def inverse_transform(self):
        pass

class ContinuousColumn(Column):

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.type = "continuous"
        self.is_fitted = False

    def fit(self, values: np.ndarray):
        self.mean = np.nanmean(values)
        self.std = np.nanstd(values)
        self.is_fitted = True

    def transform(self, values: np.ndarray):
        if not self.is_fitted:
            raise Exception("This object is not trained")
        return ((values - self.mean) / self.std)
    
    def inverse_transform(self, values: np.ndarray):
        if not self.is_fitted:
            raise Exception("This object is not trained")
        return values * self.std + self.mean
    
class DiscreteColumn(Column):

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.type = "discrete"
        self.is_fitted = False

    def fit(self, values: np.ndarray):
        self.xmax = np.nanmax(values)
        self.xmin = np.nanmin(values)
        self.is_fitted = True

    def transform(self, values: np.ndarray):
        if not self.is_fitted:
            raise Exception("This object is not trained")
        return ((values - self.xmin) / (self.xmax - self.xmin))
    
    def inverse_transform(self, values: np.ndarray):
        if not self.is_fitted:
            raise Exception("This object is not trained")
        return np.array((values * (self.xmax - self.xmin) + self.xmin), dtype=int)


class Scaler():

    def __init__(self):
        self.is_fitted = False

    def fit(self, dataframe: pd.DataFrame, continuous_columns: list, discrete_columns: list):
        cont_columns = {
            column: ContinuousColumn(column)
            for column in continuous_columns
        }
        disc_columns = {
            column: DiscreteColumn(column)
            for column in discrete_columns
        }
        self.columns = cont_columns | disc_columns
        for col_name, column in self.columns.items():
            column.fit(dataframe.loc[:, col_name].values)
        self.is_fitted = True

    def transform(self, data: pd.DataFrame):
        if not self.is_fitted:
            raise Exception("The scaler is not trained")
        transformed_data = pd.DataFrame()
        for data_column in data.columns:
            column = self.columns.get(data_column, None)
            if not column:
                warnings.warn(f"The column {data_column} was not fitted")
            transformed = column.transform(data.loc[:, data_column].values)
            transformed_data[data_column] = transformed
        return transformed_data
    
    def inverse_transform(self, data: pd.DataFrame):
        if not self.is_fitted:
            raise Exception("The scaler is not trained")
        transformed_data = pd.DataFrame()
        for data_column in data.columns:
            column = self.columns.get(data_column, None)
            if not column:
                warnings.warn(f"The column {data_column} was not fitted")
            transformed = column.inverse_transform(data.loc[:, data_column].values)
            transformed_data[data_column] = transformed
        return transformed_data


if __name__ == "__main__":

    dataframe = pd.DataFrame({
        "disc_1": [0, 1, 0, 1, 0, 0, 1, 1],
        "cont_1": [1.2, 2.5, 8.9, 6.5, 3.0, 7.4, 2.1, 0.2],
        "disc_2": [-1, 5, 10, 4, -9, 55, 1, 6],
        "cont_2": [-1.2, 4.5, 6.6, 6.1, 4.3, 7.0, 11.1, -22.2]
    })

    scaler = Scaler()

    scaler.fit(dataframe, continuous_columns=["cont_1", "cont_2"], discrete_columns=["disc_1", "disc_2"])

    transformed = scaler.transform(dataframe)

    reverse = scaler.inverse_transform(transformed)
    
    pass


    


