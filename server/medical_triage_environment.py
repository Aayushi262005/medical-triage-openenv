import json
import os
from ..models import MedicalTriageObservation, MedicalTriageAction, MedicalTriageState, StepResult

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

        self._steps = 0

    def state(self) -> MedicalTriageState:
        return self.state_data

    def reset(self) -> MedicalTriageObservation:
        self.state_data.current_patient_idx = 0
        self.state_data.is_done = False
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
        predicted_level = int(action.priority_level)

        diff = abs(predicted_level - correct_level)

        # ✅ SAFE REWARD (STRICTLY BETWEEN 0.01–0.99)
        if diff == 0:
            reward = 0.99
        elif diff == 1:
            reward = 0.75
        elif diff == 2:
            reward = 0.5
        else:
            reward = 0.25

        status_msg = "Standard"

        # 🚨 Critical under-triage
        if correct_level <= 2 and predicted_level >= 4:
            reward = 0.01
            status_msg = "CRITICAL_MISS"

        # ⚠️ Over-triage
        elif correct_level >= 4 and predicted_level <= 2:
            reward = max(reward - 0.1, 0.01)
            status_msg = "OVER_TRIAGE"

        # slight penalty for large deviation
        if diff >= 3:
            reward = max(reward - 0.05, 0.01)

        # FINAL CLAMP
        reward = max(0.01, min(0.99, reward))

        self._steps += 1
        self.state_data.current_patient_idx += 1

        done = self.state_data.current_patient_idx >= len(self.patients)
        self.state_data.is_done = done

        observation = self._get_obs() if not done else None

        info = {
            "predicted_level": predicted_level,
            "correct_level": correct_level,
            "is_critical": correct_level <= 2,
            "difference": diff,
            "status": status_msg,
            "step_reward": reward
        }

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )