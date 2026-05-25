from typing import Tuple
from datetime import datetime
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    datetime_val: datetime = Field(..., description="ISO 8601 timestamp of the event")
    init_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of origin")
    dest_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of destination")
    category: str = Field(..., description="Activity category (e.g. dinner/drinks)")

class FeedbackRequest(BaseModel):
    meeting_location: str = Field(..., description="Address of the meeting location")
    meeting_datetime: datetime = Field(..., description="ISO 8601 timestamp of meeting")
    init_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of origin")
    meeting_latlon: Tuple[float, float] = Field(..., description="(latitude, longitude) of meetup")
    category_id: str = Field(..., description="The categorical ID of the activity")
    pred_min: float = Field(..., description="Predicted duration in minutes")
    arrived_datetime: datetime = Field(..., description="ISO 8601 timestamp of actual arrival")