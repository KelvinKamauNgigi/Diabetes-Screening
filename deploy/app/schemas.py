from __future__ import annotations
from typing import Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

Education = Literal["No formal schooling", "Primary school incomplete", "Primary school completed", "Secondary school incomplete",
                    "Secondary school completed", "A-Level completed", "College/University completed", "Post-graduate degree"]
Marital = Literal["Never married", "Currently married", "Cohabiting", "Separated", "Divorced", "Widowed"]


class Patient(BaseModel):
    """One adult to score. Only age, sex and residence are required; every other answer improves the estimate."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "age": 54, "sex": "female", "residence": "urban", "height_cm": 158, "weight_kg": 82, "waist_cm": 102, "hip_cm": 111,
        "systolic_bp": 146, "diastolic_bp": 92, "heart_rate": 84, "on_bp_medication": False, "bp_ever_measured": True, "told_raised_bp": True}]})
    age: int = Field(..., ge=16, le=90, description="Age in years (the model was trained on ages 16-70)")
    sex: Literal["female", "male"]
    residence: Literal["urban", "rural"]
    height_cm: Optional[float] = Field(None, ge=100, le=220)
    weight_kg: Optional[float] = Field(None, ge=25, le=250)
    waist_cm: Optional[float] = Field(None, ge=40, le=200)
    hip_cm: Optional[float] = Field(None, ge=50, le=200)
    systolic_bp: Optional[float] = Field(None, ge=60, le=270, description="mmHg; use the average of readings 2 and 3 (STEPS protocol)")
    diastolic_bp: Optional[float] = Field(None, ge=30, le=150, description="mmHg; same averaging")
    heart_rate: Optional[float] = Field(None, ge=30, le=220, description="beats per minute at rest")
    on_bp_medication: Optional[bool] = Field(None, description="Taking blood-pressure medication in the past 2 weeks")
    bp_ever_measured: Optional[bool] = None
    told_raised_bp: Optional[bool] = Field(None, description="Ever told by a health worker that blood pressure is raised")
    cholesterol_ever_measured: Optional[bool] = None
    history_heart_attack_or_stroke: Optional[bool] = None
    current_smoker: Optional[bool] = None
    ever_drank_alcohol: Optional[bool] = None
    years_schooling: Optional[int] = Field(None, ge=0, le=25)
    education: Optional[Education] = None
    marital_status: Optional[Marital] = None
    extra_items: dict[str, Union[bool, float, str]] = Field(
        default_factory=dict, description="Any other STEPS questionnaire item by code, e.g. {\"D8\": 3, \"P1\": true}. See GET /schema for the allowed codes.")

    @model_validator(mode="after")
    def _consistency(self):
        if self.systolic_bp is not None and self.diastolic_bp is not None and self.diastolic_bp >= self.systolic_bp:
            raise ValueError("diastolic_bp must be lower than systolic_bp")
        if self.bp_ever_measured is False and self.told_raised_bp:
            raise ValueError("told_raised_bp cannot be true when bp_ever_measured is false")
        return self


class Drivers(BaseModel):
    increase: list[dict]; decrease: list[dict]

class Prediction(BaseModel):
    risk_probability: float
    risk_percent: float
    decision: Literal["refer_for_confirmatory_test", "below_referral_threshold"]
    referral_threshold: float
    times_average_risk: float
    percentile_vs_survey: float
    measurements_supplied: str
    reliability: Literal["low", "moderate", "higher"]
    derived: dict
    drivers: Drivers
    warnings: list[str]
    model: dict
    disclaimer: str

class BatchRequest(BaseModel):
    patients: list[Patient] = Field(..., min_length=1, max_length=200)
