import os
import json
import argparse
from openai import OpenAI

from server.medical_triage_environment import MedicalTriageEnvironment
from models import MedicalTriageAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

client = None
if HF_TOKEN:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception:
        client = None

def get_action(obs):
   
    if client:
        try:
            prompt = f"""
Patient: {getattr(obs, 'patient_description', '')}
Vitals: BP {getattr(obs, 'vitals_bp', '')}, HR {getattr(obs, 'vitals_hr', '')}

Return JSON:
{{
 "priority_level": 1-5,
 "reasoning": "short"
}}
"""
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            pass

    try:
        text = str(getattr(obs, "patient_description", "")).lower()

        hr_val = getattr(obs, "vitals_hr", 0)
        try:
            hr = int(hr_val)
        except Exception:
            hr = 0

        if "chest pain" in text or "unconscious" in text or hr > 140:
            return {"priority_level": 1, "reasoning": "critical"}
        elif "accident" in text or "bleeding" in text:
            return {"priority_level": 2, "reasoning": "high risk"}
        elif "fever" in text or "cold" in text:
            return {"priority_level": 4, "reasoning": "mild"}
        elif "checkup" in text:
            return {"priority_level": 5, "reasoning": "routine"}
    except Exception:
        pass

    return {"priority_level": 3, "reasoning": "fallback"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", dest="task_id", type=str, default=os.getenv("TASK_ID", "triage_basic"))
    parser.add_argument("--task_id", type=str, dest="task_id_alt", default=None)

    args, _ = parser.parse_known_args()
    task_name = args.task_id_alt if args.task_id_alt else args.task_id
    env_name = "medical_triage"

    rewards = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env={env_name} model={MODEL_NAME}", flush=True)

    try:
        env = MedicalTriageEnvironment(task_id=task_name)
        obs = env.reset()

        while obs is not None and steps < 50:
            steps += 1

            model_output = get_action(obs)

            level = model_output.get("priority_level", 3)
            if not isinstance(level, int) or not (1 <= level <= 5):
                level = 3

            error_msg = "null"

            try:
                action = MedicalTriageAction(
                    priority_level=level,
                    reasoning=str(model_output.get("reasoning", ""))
                )

                result = env.step(action)

                reward = float(getattr(result, "reward", 0.0))
                done = bool(getattr(result, "done", True))

                reward = max(0.0, min(1.0, reward))

                err = getattr(result, "last_action_error", None)
                if err:
                    error_msg = str(err)

            except Exception as e:
                reward = 0.0
                done = True
                error_msg = str(e)
                result = None

            rewards.append(f"{reward:.2f}")

            action_str = f"priority({level})"

            print(
                f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}",
                flush=True
            )

            if done:
                if reward >= 1.0:
                    success = True
                break

            obs = getattr(result, "observation", None)

        try:
            env.close()
        except Exception:
            pass

    except Exception as e:
        print(
            f"[STEP] step={steps} action=none reward=0.00 done=true error={str(e)}",
            flush=True
        )


    if not rewards:
        rewards = ["0.00"]

    rewards = [f"{float(r):.2f}" for r in rewards]

    if success:
        rewards[-1] = "1.00"

    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={','.join(rewards)}",
        flush=True
    )


if __name__ == "__main__":
    main()