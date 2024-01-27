import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeCountPipeline  # , titanic_pipe
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)
    # print(data.info())
    # print(f"Before training info\n-->{data.head()}")
    # print(f"Type of data before training\n{type(data)}")
    # print(f"Shape of data before training\n{data.shape}")

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )

    # Pipeline fitting
    bikeCountPipeline.fit(X_train, y_train)
    print("------pipeline has been trained------------")
    # print(X_test.head(2))
    y_pred = bikeCountPipeline.predict(X_test)
    print("R2 score:", r2_score(y_test, y_pred))
    print("Mean squared error:", mean_squared_error(y_test, y_pred))
    # print("Accuracy(in %):", accuracy_score(y_test, y_pred)*100)

    # persist trained model
    save_pipeline(pipeline_to_persist=bikeCountPipeline)
    # printing the score


if __name__ == "__main__":
    run_training()
