import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import *


bikeCountPipeline = Pipeline(
    [
        ("weekday_imputer", WeekdayImputer(variable=config.model_config.weekday_var)),
        (
            "weathersit_imputer",
            WeathersitImputer(variable=config.model_config.weathersit_var),
        ),
        ##==========Mapper======##
        ("map_yr", Mapper(config.model_config.yr_var, config.model_config.yr_mappings)),
        (
            "map_mnth",
            Mapper(config.model_config.mnth_var, config.model_config.mnth_mappings),
        ),
        (
            "map_season",
            Mapper(config.model_config.season_var, config.model_config.season_mappings),
        ),
        (
            "map_weathersit",
            Mapper(
                config.model_config.weathersit_var,
                config.model_config.weathersit_mappings,
            ),
        ),
        (
            "map_holiday",
            Mapper(
                config.model_config.holiday_var, config.model_config.holiday_mappings
            ),
        ),
        (
            "map_workingday",
            Mapper(
                config.model_config.workingday_var,
                config.model_config.workingday_mappings,
            ),
        ),
        ("map_hr", Mapper(config.model_config.hr_var, config.model_config.hr_mappings)),
        # Transformation of age column
        ("outlier_handler_hum", OutlierHandler(config.model_config.hum_var)),
        (
            "outlier_handler_windspeed",
            OutlierHandler(config.model_config.windspeed_var),
        ),
        # One hot encoding weekday
        (
            "weekDayOneHotEncoding",
            WeekdayOneHotEncoder(config.model_config.weekday_var),
        ),
        # Drop columns
        ("dropColumns", DropColumns(["dteday", "casual", "registered", "weekday"])),
        # scale
        ("scaler", StandardScaler()),
        (
            "model_rf",
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        ),
    ]
)
