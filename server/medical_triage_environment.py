import json
import os
from models import MedicalTriageObservation, MedicalTriageAction, MedicalTriageState, StepResult  # ✅ added StepResult

try:
    import openenv
    BaseEnvironment = getattr(openenv, 'BaseEnv', getattr(openenv, 'BaseEnvironment', object))
except ImportError:
    BaseEnvironment = object 

class MedicalTriageEnvironment(BaseEnvironment):
    def __init__(self, task_id="triage_basic"):
        self.task_id = task_id
        self.state_data = MedicalTriageState(current_patient_idx=0, is_done=False)
        current_dir = os.path.dirname(__file__)
        path = os.path.join(current_dir, '..', 'data', f'{task_id}.json')
        with open(path, "r") as f:
            self.patients = json.load(f)
        self._total_reward = 0.0
        self._steps = 0

    def state(self) -> MedicalTriageState:
        return self.state_data

    def reset(self) -> MedicalTriageObservation:
        self.state_data.current_patient_idx = 0
        self.state_data.is_done = False
        self._total_reward = 0.0
        self._steps = 0
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
        correct_level = int(current_p["correct"])
        given_level = int(action.priority_level)

        reward = 0.0
        diff = abs(given_level - correct_level)
        status_msg = "Standard Triage"

        if diff == 0:
            reward = 1.0
        elif diff == 1:
            reward = 0.5
        
        if correct_level <= 2 and given_level >= 4:
            reward = -2.0
            status_msg = "CRITICAL SAFETY VIOLATION"
        elif correct_level >= 4 and given_level <= 2:
            reward -= 0.5
            status_msg = "Resource Over-utilization"

        self._total_reward += reward
        self._steps += 1
        self.state_data.current_patient_idx += 1

        done = self.state_data.current_patient_idx >= len(self.patients)
        self.state_data.is_done = done

        max_possible = float(self._steps)
        grading_score = max(0.0, min(1.0, self._total_reward / max_possible)) if max_possible > 0 else 0.0

        observation = self._get_obs() if not done else None
        
        info = {
            "status": status_msg,
            "correct": correct_level,
            "grading_score": grading_score,
            "step_reward": reward
        }

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )