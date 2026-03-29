---
title: Medical Triage Environment
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# 🏥 Clinical Medical Triage Environment (OpenEnv)

## 🌟 Motivation

In emergency departments, the first few minutes of patient assessment are critical. This environment simulates a **clinical triage task**, where an AI agent acts as a triage nurse responsible for prioritizing patients on a scale from 1 (critical) to 5 (non-urgent), based on symptom descriptions and vital signs such as heart rate and blood pressure.

The goal is to evaluate the **clinical reasoning, decision-making, and safety alignment** of AI agents in high-stakes scenarios.

---

## 🛠️ Environment Specification

### 📝 Observation Space

At each step, the agent receives a `MedicalTriageObservation` object:

| Field | Type | Description |
|------|------|------------|
| `patient_description` | `str` | Narrative describing symptoms, history, and presentation |
| `vitals_hr` | `int` | Heart rate in beats per minute (BPM) |
| `vitals_bp` | `str` | Blood pressure reading (e.g., "160/100") |

---

### ⚡ Action Space

The agent must return a `MedicalTriageAction` JSON object:

- **`priority_level`** (`int`): Priority from 1 (critical) to 5 (non-urgent)  
- **`reasoning`** (`str`): Clinical justification for the assigned priority  

---

## 🎯 Task Descriptions

- **triage_basic (Easy):**  
  Straightforward cases with clear symptoms (e.g., minor injuries vs. fractures).

- **triage_vitals (Medium):**  
  Requires interpreting vital signs such as tachycardia or hypertension.

- **triage_emergency (Hard):**  
  Complex, high-stakes scenarios where subtle symptoms may indicate life-threatening conditions.

---

## 🏆 Reward Design & Safety

The reward function is designed to prioritize patient safety and clinical correctness:

- **Correct Priority:** +1.0  
- **Close Match (±1 level):** +0.5  
- **Critical Safety Penalty:** -2.0  
  - Applied when a life-threatening patient (Level 1) is assigned Level 3 or higher  
- **Resource Misallocation Penalty:** -0.5  
  - Applied when non-urgent cases are incorrectly assigned Level 1  

---

## 🚀 Setup & Usage

### 1. Build the Docker Image

This step packages the environment, server, and dependencies:

```bash
docker build -t medical-triage:latest .
```

---

### 2. Run the Environment Server

Start the OpenEnv server. This exposes the environment on port `7860`:

```bash
docker run -p 7860:7860 medical-triage:latest
```

---

### 3. Run the Baseline Inference Agent

Open a separate terminal window and run the evaluation agent.

#### 📌 Windows (PowerShell)

```powershell
$env:HF_TOKEN="your_huggingface_token_here"
$env:MODEL_NAME="gpt-4o"
python inference.py
```

#### 📌 Mac/Linux (Bash)

```bash
export HF_TOKEN="your_huggingface_token_here"
export MODEL_NAME="gpt-4o"
python inference.py
```

---

## 📂 Project Structure

```plaintext
medical_triage/
├── data/
│   ├── triage_basic.json
│   ├── triage_vitals.json
│   └── triage_emergency.json
├── server/
│   ├── __init__.py
│   ├── app.py
│   └── medical_triage_environment.py
├── models.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── inference.py
└── README.md
```