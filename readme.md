README — Traffic Flow Optimiser Agent
📌 Overview

The Traffic Flow Optimiser Agent is an AI-based traffic analysis system that supports two modes:

Simulation Mode — runs a LangGraph workflow to generate synthetic congestion results.

Dataset Mode — loads and processes traffic_data.csv to give real-world congestion analytics.

Both modes are accessed through a single HTTP API.
The project also includes a minimal frontend UI to send requests easily.

📂 Project Structure
/project-root
│── api.py                     ← FastAPI backend
│── graph.py                   ← LangGraph simulation workflow
│── dataset_loader.py          ← Dataset mode processing
│── traffic_data.csv           ← Dataset used in dataset mode
│── frontend/                  ← Simple HTML frontend UI
│     └── index.html
│── requirements.txt
│── Dockerfile (optional for deployment)

🛠 Installation
1. Install all dependencies
pip install -r requirements.txt

🚀 Running the Backend

Start FastAPI:

uvicorn api:app --host 0.0.0.0 --port 8000 --reload


Frontend will run at:

http://127.0.0.1:8000




Backend will run at (interactive):

http://127.0.0.1:8000/docs

Example Request 
{
  "messages": [
    {
      "role": "assistant",
      "content": "Hello! I can help you simulate traffic under different conditions."
    },
    {
      "role": "user",
      "content": {
        "layman_description": "Simulate rush hour at A,B and D.",
        "mode": "simulation",
        "scenario": "rain",
        "intersections": ["A","B", "D"]
      }
    }
  ]
}
