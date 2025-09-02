
from typing import List, Annotated
from pydantic import BaseModel, Field

FeatureVector = Annotated[List[float], Field(min_length=4, max_length=4)]

class PredictRequest(BaseModel):
    features: List[FeatureVector]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"features": [[5.1, 3.5, 1.4, 0.2]]},
                {"features": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}
            ]
        }
    }

class PredictResponse(BaseModel):
    predictions: List[int]