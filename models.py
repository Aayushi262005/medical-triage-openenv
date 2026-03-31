from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- OBSERVATION ---
class MedicalTriageObservation(BaseModel):
    patient_description: str
    vitals_bp: str
    vitals_hr: int
    current_waiting_room_count: int

# --- ACTION ---
class MedicalTriageAction(BaseModel):
    priority_level: int  # 1 to 5
    reasoning: str

# --- STATE ---
class MedicalTriageState(BaseModel):
    current_patient_idx: int
    is_done: bool

# --- STEP RESULT ---
class StepResult(BaseModel):
    observation: Optional[MedicalTriageObservation]
    reward: float
    done: bool
    info: Dict[str, Any]