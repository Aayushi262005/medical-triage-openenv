import openenv
import json
import os
from dataclasses import dataclass
from typing import Any, Optional
from models import MedicalTriageObservation, MedicalTriageAction, MedicalTriageState

@dataclass
class StepResult:
    observation: Optional[MedicalTriageObservation]
    reward: float
    done: bool
    info: dict

# Handles Base Class safely
BaseEnvironment = getattr(openenv, 'BaseEnv', getattr(openenv, 'BaseEnvironment', object))

class MedicalTriageEnvironment(BaseEnvironment):
    def __init__(self, task_id="triage_basic"):
        self.task_id = task_id
        # Standardized to self.state_data to match your reset/state methods
        self.state_data = MedicalTriageState(current_patient_idx=0, is_done=False)

        # Updated path to look into the 'data' folder for specific tasks
        current_dir = os.path.dirname(__file__)
        path = os.path.join(current_dir, '..', 'data', f'{task_id}.json')
        
        with open(path, "r") as f:
            self.patients = json.load(f)

    def state(self) -> MedicalTriageState:
        return self.state_data

    def reset(self) -> MedicalTriageObservation:
        self.state_data.current_patient_idx = 0
        self.state_data.is_done = False
        return self._get_obs()
    
    def _get_obs(self):
        if self.state_data.current_patient_idx >= len(self.patients):
            return None
        p = self.patients[self.state_data.current_patient_idx]
        return MedicalTriageObservation(
            patient_description=p["desc"],
            vitals_hr=p["hr"],
            vitals_bp=p["bp"],
            current_waiting_room_count=len(self.patients) - self.state_data.current_patient_idx
        )

    def step(self, action: MedicalTriageAction):
        current_p = self.patients[self.state_data.current_patient_idx]
        correct_level = current_p["correct"]
        given_level = action.priority_level

        reward = 0.0
        diff = abs(given_level - correct_level)
        
        if diff == 0:
            reward = 1.0
        elif diff == 1:
            reward = 0.5
            
        info_msg = "Standard Triage"
        if correct_level == 1 and given_level >= 3:
            reward -= 2.0
            info_msg = "CRITICAL SAFETY VIOLATION"
        elif correct_level >= 4 and given_level == 1:
            reward -= 0.5
            info_msg = "Resource Over-utilization"

        self.state_data.current_patient_idx += 1
        done = self.state_data.current_patient_idx >= len(self.patients)
        self.state_data.is_done = done

        # 4. RETURN AS A TUPLE
        observation = self._get_obs() if not done else None
        info = {"status": info_msg, "correct": correct_level}
        
        return observation, reward, done, info