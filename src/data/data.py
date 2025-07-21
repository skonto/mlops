from typing import List

from pydantic import BaseModel, field_validator


class BatchInput(BaseModel):
    features: List[List[float]]  # Batch of input samples

    @field_validator("features")
    @classmethod
    def validate_shape(cls, features):
        for i, sample in enumerate(features):
            if len(sample) != 4:
                raise ValueError(f"Sample at index {i} must have exactly 4 features. Sample: {sample}")
        return features
