from classification_model.config import config

import pandas as pd


def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()

    # check for features with NA not seen during training
    if input_data[config.FEATURES_VALIDATE].isnull().any().any():
        validated_data = validated_data.dropna(
            axis=0, subset=config.FEATURES_VALIDATE
        )

    return validated_data
