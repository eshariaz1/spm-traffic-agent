# api.py
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Optional, Literal, Union
import time
import logging
import json

from graph import traffic_app
from dataset_loader import load_dataset, filter_dataset, aggregate_rows
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os


# -----------------------------
# App + Logging Setup
# -----------------------------
app = FastAPI(title="Traffic Flow Optimiser Agent API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("traffic_agent")

start_time = time.time()
request_count = 0
last_scenario: Optional[str] = None
last_run_timestamp: Optional[int] = None
last_explanation: Optional[str] = None
last_report: Optional[str] = None

feedback_store: List[Dict[str, Any]] = []

# cache dataset in memory
_dataset_df = None


def get_dataset_df():
    global _dataset_df
    if _dataset_df is None:
        _dataset_df = load_dataset("traffic_data.csv")
        logger.info(f"Loaded dataset with {_dataset_df.shape[0]} rows.")
    return _dataset_df


# -----------------------------
# Core Traffic Request
# -----------------------------
class TrafficRequest(BaseModel):
    # 🔥 now supports BOTH modes
    mode: Literal["simulation", "dataset"] = "simulation"

    scenario: Literal["rush_hour", "accident", "rain", "event_day", "off_peak"] = (
        "rush_hour"
    )

    intersections: List[str] = Field(
        default_factory=lambda: ["A", "B", "C", "D"],
        description="Non-empty list of intersections.",
    )

    @model_validator(mode="after")
    def validate_intersections(self) -> "TrafficRequest":
        if not self.intersections:
            raise ValueError("intersections must be a non-empty list")

        cleaned: List[str] = []
        for item in self.intersections:
            if not isinstance(item, str):
                raise ValueError("each intersection must be a string")

            s = item.strip()
            if not s:
                raise ValueError("intersection identifiers cannot be blank")

            cleaned.append(s)

        # 🔥 MAX 4 INTERSECTIONS — RULE
        if len(cleaned) > 4:
            raise ValueError(
                "A maximum of 4 intersections is supported. "
                "Please provide between 1 and 4 intersections only."
            )

        self.intersections = cleaned
        return self


# -----------------------------
# Feedback Request
# -----------------------------
class FeedbackRequest(BaseModel):
    user: Optional[str] = None
    scenario: Optional[
        Literal["rush_hour", "accident", "rain", "event_day", "off_peak"]
    ] = None
    intersections: Optional[List[str]] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    comment: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_fields(self) -> "FeedbackRequest":
        if self.user is not None:
            self.user = self.user.strip() or None

        if self.intersections is not None:
            cleaned: List[str] = []
            for item in self.intersections:
                if not isinstance(item, str):
                    raise ValueError("each intersection in feedback must be a string")
                s = item.strip()
                if not s:
                    raise ValueError("intersection identifiers cannot be blank")
                cleaned.append(s)

            if len(cleaned) > 4:
                raise ValueError(
                    "A maximum of 4 intersections is supported. "
                    "Please provide between 1 and 4 intersections only."
                )

            self.intersections = cleaned

        self.comment = self.comment.strip()
        if not self.comment:
            raise ValueError("comment cannot be empty")

        return self


# -----------------------------
# Chat Models
# -----------------------------
class ChatMessage(BaseModel):
    role: Literal["assistant", "user", "system"]
    content: Union[str, Dict[str, Any]]


class ChatTrafficRequest(BaseModel):
    messages: List[ChatMessage]
    # 🔥 both modes for chat as well
    mode: Optional[Literal["simulation", "dataset"]] = "simulation"
    scenario: Optional[
        Literal["rush_hour", "accident", "rain", "event_day", "off_peak"]
    ] = "rush_hour"
    intersections: Optional[List[str]] = None

    @model_validator(mode="after")
    def validate_messages(self) -> "ChatTrafficRequest":
        if not self.messages:
            raise ValueError("messages must be a non-empty list")

        user_msgs = [m for m in self.messages if m.role == "user"]
        if not user_msgs:
            raise ValueError("at least one user message is required")

        return self


# -----------------------------
# Chat → TrafficRequest Converter
# -----------------------------
def chat_to_traffic_request(chat: ChatTrafficRequest) -> TrafficRequest:
    user_messages = [m for m in chat.messages if m.role == "user"]
    last_user = user_messages[-1]

    payload: Dict[str, Any] = {}

    # (1) Start from top-level
    if chat.mode is not None:
        payload["mode"] = chat.mode
    if chat.scenario is not None:
        payload["scenario"] = chat.scenario
    if chat.intersections:
        payload["intersections"] = chat.intersections

    # Merge helper
    def merge_from_dict(d: Dict[str, Any]):
        for key in ("mode", "scenario", "intersections"):
            if key in d:
                payload[key] = d[key]

    # (2) Override with last user message
    if isinstance(last_user.content, dict):
        merge_from_dict(last_user.content)
    elif isinstance(last_user.content, str):
        try:
            parsed = json.loads(last_user.content)
            if isinstance(parsed, dict):
                merge_from_dict(parsed)
        except json.JSONDecodeError:
            # layman text, ignore for structure
            pass

    # Enforce intersection limit *before* building TrafficRequest
    if "intersections" in payload:
        if not isinstance(payload["intersections"], list):
            raise ValueError("Intersections must be a list of strings.")
        if len(payload["intersections"]) > 4:
            raise ValueError(
                "A maximum of 4 intersections is supported. "
                "Please provide between 1 and 4 intersections only."
            )

    return TrafficRequest(**payload)


# -----------------------------
# Dataset-mode intelligence helper
# -----------------------------
def build_dataset_response(request: TrafficRequest) -> Dict[str, Any]:
    """
    Use the CSV dataset, compute data-driven congestion levels,
    and return the SAME shape as the LangGraph simulation response.
    """

    df = filter_dataset(
        get_dataset_df(), request.scenario, request.intersections
    )

    stats = aggregate_rows(df)
    if stats is None:
        raise ValueError(
            "No matching records found in dataset for the given scenario/intersections."
        )

    # ---- Data-driven thresholds based on dataset stats ----
    total_volume = float(stats["total_volume"])
    avg_queue = float(stats["average_queue_length"])
    avg_speed = float(stats["average_speed"])
    num_records = int(stats["num_records"])
    max_queue = float(stats["max_queue_length"])

    avg_volume_per_record = total_volume / max(num_records, 1)

    # Decide congestion level from dataset (relative thresholds)
    if avg_queue >= 45 or avg_volume_per_record >= 95:
        overall_level = "severe"
    elif avg_queue >= 35 or avg_volume_per_record >= 80:
        overall_level = "high"
    elif avg_queue >= 20 or avg_volume_per_record >= 60:
        overall_level = "medium"
    else:
        overall_level = "low"

    # Per-intersection view: split total volume across intersections
    n_int = max(len(request.intersections), 1)
    per_int_volume = total_volume / n_int

    congestion_levels: Dict[str, Any] = {}
    predicted: Dict[str, Any] = {}
    hotspots: List[str] = []

    for name in request.intersections:
        level = overall_level
        congestion_levels[name] = {
            "volume": int(round(per_int_volume)),
            "queue_length": float(avg_queue),
            "avg_speed": float(avg_speed),
            "level": level,
        }

        # trend based on level + speed
        if level in ("high", "severe") and avg_speed < 25:
            trend = "likely_increase"
        elif level in ("medium", "high"):
            trend = "stable"
        else:
            trend = "likely_decrease"

        predicted[name] = {
            "current_level": level,
            "trend": trend,
        }

        if level in ("high", "severe"):
            hotspots.append(name)

    congestion = {
        "levels": congestion_levels,
        "hotspots": hotspots,
    }

    metrics = {
        "total_volume": int(round(total_volume)),
        "average_queue_length": float(avg_queue),
        "num_intersections": len(request.intersections),
        "average_speed": float(avg_speed),
        "max_queue_length": int(round(max_queue)),
    }

    # Weather / context based on scenario
    if request.scenario == "rain":
        weather = {"type": "rain", "severity": "dataset_observed"}
    elif request.scenario == "event_day":
        weather = {"type": "event_day", "severity": "dataset_observed"}
    elif request.scenario == "accident":
        weather = {"type": "accident", "severity": "dataset_observed"}
    else:
        weather = {"type": request.scenario, "severity": "dataset_observed"}

    # Incident probability from congestion + queue length
    base_probs = {"low": 0.05, "medium": 0.15, "high": 0.30, "severe": 0.5}
    incident_prob = base_probs.get(overall_level, 0.1)
    if avg_queue > 45:
        incident_prob += 0.1
    incident_prob = max(0.0, min(incident_prob, 0.95))

    # Recommendations
    recommendations: List[str] = []
    if hotspots:
        recommendations.append(
            f"Based on dataset trends, allocate extra green time to {', '.join(hotspots)} during {request.scenario}."
        )

    if overall_level in ("high", "severe"):
        recommendations.append(
            "Use trained personnel or traffic wardens near the busiest junctions during peak intervals."
        )

    if not recommendations:
        recommendations.append(
            "Dataset suggests current timing plan is sufficient; continue monitoring queues and speeds."
        )

    # Explanation for teacher/demo
    explanation = (
        f"Using the historical dataset for scenario '{request.scenario}', "
        f"the average volume per record is {avg_volume_per_record:.1f} vehicles with an "
        f"average queue length of {avg_queue:.1f} vehicles and average speed {avg_speed:.1f} km/h. "
        f"From these dataset-derived statistics, overall congestion is classified as {overall_level}. "
        f"The same congestion level is projected across intersections "
        f"{', '.join(request.intersections)}, and predicted trends are computed from dataset behaviour "
        f"(higher queues + lower speeds → likely increase)."
    )

    raw_response = {
        "agent_name": "Traffic Flow Optimiser Agent",
        "scenario": request.scenario,
        "signal_plan": {
            # simple plan based on congestion level
            name: {
                "green_seconds": 70 if overall_level == "severe"
                else 60 if overall_level == "high"
                else 45 if overall_level == "medium"
                else 30,
                "red_seconds": 90
                - (
                    70 if overall_level == "severe"
                    else 60 if overall_level == "high"
                    else 45 if overall_level == "medium"
                    else 30
                ),
            }
            for name in request.intersections
        },
        "congestion": congestion,
        "metrics": metrics,
        "predicted_congestion": predicted,
        "weather_severity": weather,
        "incident_probability": round(incident_prob, 2),
        "recommendations": recommendations,
        "explanation": explanation,
        "status": "ok",
    }

    return {
        "message": f"Dataset analysis for scenario '{request.scenario}' completed.",
        "raw_response": raw_response,
    }
# -----------------------------
# Dataset Markdown Report Builder
# -----------------------------
def build_dataset_markdown_report(raw_response: Dict[str, Any]) -> str:
    scenario = raw_response.get("scenario")
    metrics = raw_response.get("metrics", {})
    congestion = raw_response.get("congestion", {})
    predicted = raw_response.get("predicted_congestion", {})
    hotspots = congestion.get("hotspots", [])
    signal_plan = raw_response.get("signal_plan", {})
    incident_prob = raw_response.get("incident_probability")
    explanation = raw_response.get("explanation", "")
    
    report = f"""
========================
 DATASET ANALYSIS REPORT
========================

Scenario: **{scenario}**

---
### 🔹 Metrics (Dataset Derived)
- Total Volume: {metrics.get("total_volume")}
- Avg Queue Length: {metrics.get("average_queue_length")}
- Avg Speed: {metrics.get("average_speed")}
- Max Queue Length: {metrics.get("max_queue_length")}
- Intersections Analyzed: {metrics.get("num_intersections")}

---
### 🔹 Congestion Levels (From Dataset)
"""
    for inter, data in congestion.get("levels", {}).items():
        report += f"- **{inter}** → {data['level']} (queue: {data['queue_length']}, speed: {data['avg_speed']})\n"

    report += "\n---\n### 🔹 Predicted Trends\n"
    for inter, data in predicted.items():
        report += f"- **{inter}** → now: {data['current_level']}, trend: {data['trend']}\n"

    report += "\n---\n### 🔹 Signal Timing Suggestions\n"
    for inter, plan in signal_plan.items():
        report += f"- **{inter}** → green: {plan['green_seconds']}s, red: {plan['red_seconds']}s\n"

    report += f"""
---
### 🔹 Incident Probability
Estimated probability: **{incident_prob * 100:.1f}%**

---
### 🔹 Hotspots
{", ".join(hotspots) if hotspots else "No major hotspots detected."}

---
### 🔹 Explanation 
{explanation}

"""
    return report.strip()



# -----------------------------
# Global Error Handler
# -----------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error while processing {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error in Traffic Flow Optimiser Agent.",
            "hint": "Check request format and agent logs for details.",
        },
    )


# -----------------------------
# Main Agent Endpoint
# -----------------------------
@app.post("/api/traffic-agent")
def run_traffic_agent(chat_request: ChatTrafficRequest):
    global request_count, last_scenario, last_run_timestamp, last_explanation, last_report

    
    try:
        request = chat_to_traffic_request(chat_request)

        request_count += 1
        last_scenario = request.scenario
        last_run_timestamp = int(time.time())

        # ----------------- MODE: DATASET -----------------
        if request.mode == "dataset":
            logger.info(
                f"Running DATASET mode for scenario={request.scenario}, intersections={request.intersections}"
            )

            # Build dataset response
            dataset_result = build_dataset_response(request)
            last_explanation = dataset_result["raw_response"]["explanation"]

            # Build dataset markdown report
            dataset_raw = dataset_result["raw_response"]
            dataset_markdown = build_dataset_markdown_report(dataset_raw)
            last_report = dataset_markdown

            return {
                "agent_name": "traffic-agent",
                "status": "success",
                "data": dataset_result,
                "error_message": None,
            }


        # ----------------- MODE: SIMULATION (existing) -----------------
        logger.info(
            f"Running SIMULATION mode for scenario={request.scenario}, intersections={request.intersections}"
        )

        initial_state = {"raw_request": request.dict()}
        result_state = traffic_app.invoke(initial_state)

        graph_response = result_state["api_response"]
        last_explanation = graph_response.get("explanation", "")

        report = result_state.get("markdown_report")
        last_report = report.strip() if isinstance(report, str) and report.strip() else None

        return {
            "agent_name": "traffic-agent",
            "status": "success",
            "data": {
                "message": last_explanation or "Traffic agent run completed.",
                "raw_response": graph_response,
            },
            "error_message": None,
        }

    except Exception as exc:
        logger.exception("Error generating agent response")
        return {
            "agent_name": "traffic-agent",
            "status": "error",
            "data": None,
            "error_message": str(exc),
        }


# -----------------------------
# Report Endpoint
# -----------------------------
@app.get("/report", response_class=PlainTextResponse)
def get_latest_report():
    if not last_report:
        return PlainTextResponse(
            "No report available yet. Please call /api/traffic-agent (simulation mode) at least once.",
            status_code=404,
        )

    health = {
        "status": "ok",
        "uptime_seconds": int(time.time() - start_time),
        "agent": "Traffic Flow Optimiser Agent",
    }

    status = {
        "requests_served": request_count,
        "last_scenario": last_scenario,
        "last_explanation": last_explanation,
        "timestamp": last_run_timestamp,
    }

    combined = f"""
============================================================
                TRAFFIC FLOW OPTIMISER — FULL REPORT
============================================================
========================
  🔹 Agent Health
========================
Status           : {health['status']}
Uptime (seconds) : {health['uptime_seconds']}
Agent Name       : {health['agent']}
========================
  🔹 System Status
========================
Total Requests Served     : {status['requests_served']}
Last Scenario Processed   : {status['last_scenario']}
Last Explanation Summary  : {status['last_explanation']}
Last Run Timestamp        : {status['timestamp']}
========================
  🔹 Traffic Analysis Report
========================
{last_report}
============================================================
                  END OF REPORT
============================================================
    """

    return PlainTextResponse(combined.strip())


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "agent": "Traffic Flow Optimiser Agent",
        "uptime_seconds": int(time.time() - start_time),
        "message": "Service is running normally.",
    }


# -----------------------------
# Status Endpoint
# -----------------------------
@app.get("/status")
def status_check():
    return {
        "agent_name": "Traffic Flow Optimiser Agent",
        "status": "operational",
        "uptime_seconds": int(time.time() - start_time),
        "total_requests_served": request_count,
        "last_scenario_processed": last_scenario,
        "last_run_timestamp": last_run_timestamp,
        "last_explanation_summary": last_explanation,
        "supported_scenarios": [
            "rush_hour",
            "accident",
            "rain",
            "event_day",
            "off_peak",
        ],
        "supported_modes": ["simulation", "dataset"],
        "feedback_entries": len(feedback_store),
    }


# -----------------------------
# Feedback
# -----------------------------
@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    entry = {
        "timestamp": int(time.time()),
        "user": feedback.user,
        "scenario": feedback.scenario,
        "intersections": feedback.intersections,
        "rating": feedback.rating,
        "comment": feedback.comment,
    }
    feedback_store.append(entry)

    return {
        "status": "received",
        "total_feedback_entries": len(feedback_store),
    }


@app.get("/feedback")
def list_feedback(limit: int = Query(10, ge=1, le=100)):
    recent = feedback_store[-limit:]
    return {"count": len(recent), "entries": recent}


# -----------------------------
# Simple Home Page (optional)
# -----------------------------
# Serve frontend directory
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
