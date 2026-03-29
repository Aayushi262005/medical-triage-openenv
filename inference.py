import os
import json
from openai import OpenAI
from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

# MANDATORY: Use the exact variable names from the problem statement
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# They specifically mentioned HF_TOKEN in the requirements
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "sk-test"
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

def run_task(task_id):
    env = MedicalTriageEnvironment(task_id=task_id)
    obs = env.reset()
    done = False
    results = []

    print(f"Starting Task: {task_id}")

    while not done and obs is not None:
        prompt = (
            f"Triage this patient: {obs.patient_description}. "
            f"Vitals: HR {obs.vitals_hr}, BP {obs.vitals_bp}. "
            f"Provide priority 1-5 and reasoning in JSON format with keys 'priority_level' and 'reasoning'."
        )
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            action_data = json.loads(content)
            action = MedicalTriageAction(**action_data)
            
            step_result = env.step(action)
            results.append(step_result)
            
            obs = step_result.observation
            done = step_result.done
            
            print(f"  Step: Priority {action.priority_level} assigned. Reward: {step_result.reward}")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            break # Exit current task loop on critical error
    
    return results

if __name__ == "__main__":
    # Ensure tasks match your openenv.yaml
    tasks = ["triage_basic", "triage_vitals", "triage_emergency"]
    for task in tasks:
        print(f"--- Running {task} ---")
        run_task(task)
    print("Inference baseline complete.")