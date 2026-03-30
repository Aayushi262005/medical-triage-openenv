import os
import json
from openai import OpenAI
from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

def run_task(task_id):
    env = MedicalTriageEnvironment(task_id=task_id)
    obs = env.reset()
    done = False
    score = 0.0
    while not done and obs is not None:
        prompt = f"Triage: {obs.patient_description}. HR: {obs.vitals_hr}, BP: {obs.vitals_bp}. JSON only: {{'priority_level': <int>, 'reasoning': <str>}}"
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        action = MedicalTriageAction(**data)
        obs, reward, done, info = env.step(action)
        score = info.get("grading_score", 0.0)
    return score

if __name__ == "__main__":
    tasks = ["triage_basic", "triage_vitals", "triage_emergency"]
    results = {t: run_task(t) for t in tasks}
    print(json.dumps(results, indent=4))