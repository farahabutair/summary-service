from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class VitalsData(BaseModel):
    BP: Optional[str] = None
    HR: Optional[str] = None
    RR: Optional[str] = None
    Temp: Optional[str] = None
    SpO2: Optional[str] = None

class Surgery(BaseModel):
    name: str
    status: str

class PatientData(BaseModel):
    Age: Optional[str] = None
    Gender: Optional[str] = None
    Diagnosis: Optional[str] = None
    Symptoms: Optional[List[str]] = []
    Medications: Optional[List[str]] = []
    Surgeries: Optional[List[Surgery]] = []
    Allergies: Optional[List[str]] = []
    Medical_Warnings: Optional[List[str]] = []
    Problems: Optional[List[str]] = []
    Vitals: Optional[VitalsData] = None
    Lab_Results: Optional[Dict[str, str]] = {}

class SummarizeRequest(BaseModel):
    request_id: Optional[str] = None
    patient_data: Optional[PatientData] = None
    text: Optional[str] = None

class SummarizeResponse(BaseModel):
    short_summary: str
    bullet_summary: List[str]
    keywords: List[str]
    confidence_note: str