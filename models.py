from pydantic import BaseModel
from typing import List, Optional

class MedicalTriageObservation(BaseModel):
    patient_description: str
    vitals_bp: str
    vitals_hr: int
    current_waiting_room_count: int

class MedicalTriageAction(BaseModel):
    priority_level: int  # 1 to 5
    reasoning: str

class MedicalTriageState(BaseModel):
    current_patient_idx: int
    is_done: bool