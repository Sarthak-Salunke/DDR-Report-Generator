"""
Data models for DDR Report Generator
Uses Pydantic for validation and type safety
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal, Dict
from datetime import datetime


class Observation(BaseModel):
    """Individual observation extracted from inspection or thermal reports"""
    
    location: str = Field(
        description="Area/room where issue observed (e.g., 'Hall', 'Master Bedroom')"
    )
    issue_type: str = Field(
        description="Type of defect (e.g., 'Dampness', 'Crack', 'Seepage')"
    )
    description: str = Field(
        description="Detailed description of the observation"
    )
    severity: Optional[Literal["Critical", "High", "Medium", "Low"]] = Field(
        default=None,
        description="Severity level of the issue"
    )
    source_document: Literal["inspection", "thermal"] = Field(
        description="Source document type"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence score of extraction (0-1)"
    )
    evidence: Optional[str] = Field(
        default=None,
        description="Photo reference, measurement, or other evidence"
    )
    
    @validator('location')
    def location_not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Location cannot be empty")
        return v.strip()
    
    @validator('issue_type')
    def issue_type_not_empty(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Issue type cannot be empty")
        return v.strip()


class ThermalReading(BaseModel):
    """Thermal imaging data point"""
    
    location: str = Field(description="Area where thermal reading taken")
    hotspot_temp: float = Field(description="Maximum temperature in °C")
    coldspot_temp: float = Field(description="Minimum temperature in °C")
    temperature_delta: float = Field(description="Temperature difference in °C")
    date: str = Field(description="Date of thermal imaging")
    image_reference: str = Field(description="Thermal image filename/number")
    interpretation: str = Field(
        description="LLM interpretation of thermal anomaly"
    )
    
    @validator('temperature_delta')
    def validate_delta(cls, v, values):
        if 'hotspot_temp' in values and 'coldspot_temp' in values:
            expected_delta = values['hotspot_temp'] - values['coldspot_temp']
            if abs(v - expected_delta) > 0.5:
                raise ValueError(f"Temperature delta mismatch. Expected ~{expected_delta}, got {v}")
        return v


class PropertyDetails(BaseModel):
    """Property information from inspection report"""
    
    property_type: str = Field(description="Type of property (e.g., Flat, House)")
    floors: Optional[int] = Field(default=None, description="Number of floors")
    inspection_date: str = Field(description="Date of inspection")
    inspected_by: List[str] = Field(
        default_factory=list,
        description="Names of inspectors"
    )
    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Overall inspection score"
    )
    flagged_items: Optional[int] = Field(
        default=None,
        description="Number of flagged items"
    )


class InspectionData(BaseModel):
    """Complete structured data from inspection report"""
    
    property_details: PropertyDetails
    observations: List[Observation]
    root_causes: List[str] = Field(
        default_factory=list,
        description="Identified root causes of issues"
    )
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class ThermalData(BaseModel):
    """Complete structured data from thermal report"""
    
    readings: List[ThermalReading]
    date: str = Field(description="Date of thermal imaging session")
    device: Optional[str] = Field(default=None, description="Thermal camera model")
    metadata: Dict = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class MergedObservation(BaseModel):
    """Observation merged from multiple sources"""
    
    location: str
    issue_type: str
    description: str
    severity: Optional[str] = None
    sources: List[Literal["inspection", "thermal"]]
    evidence: List[str] = Field(default_factory=list)
    confidence: float
    is_duplicate: bool = Field(
        default=False,
        description="Flag if merged from duplicates"
    )
    conflict_detected: bool = Field(
        default=False,
        description="Flag if conflicting information exists"
    )
    conflict_details: Optional[str] = Field(
        default=None,
        description="Details of conflict if any"
    )


class DDRReport(BaseModel):
    """Final DDR report structure"""
    
    property_details: PropertyDetails
    executive_summary: str
    observations_by_area: Dict[str, List[MergedObservation]]
    total_observations: int
    critical_issues: List[MergedObservation]
    conflicts_detected: List[Dict]
    recommendations: List[str]
    limitations: List[str]
    generated_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


# Utility Functions

def observation_to_dict(obs: Observation) -> dict:
    """Convert Observation to dictionary for processing"""
    return obs.model_dump()


def validate_extraction_completeness(data: InspectionData) -> tuple[bool, List[str]]:
    """Validate that extracted data meets minimum requirements"""
    errors = []
    
    if not data.observations:
        errors.append("No observations extracted")
    
    if not data.property_details.inspection_date:
        errors.append("Missing inspection date")
    
    if len(data.observations) < 5:
        errors.append(f"Too few observations: {len(data.observations)} (expected at least 5)")
    
    return len(errors) == 0, errors