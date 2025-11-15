# api.py
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse  # 👈 already okay
from pydantic import BaseModel, Field, model_validator
from typing import Any, Dict, List, Optional, Literal
import time
import logging

from graph import traffic_app  # import the graph we just built


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
last_report: Optional[str] = None   # 👈 NEW: stores latest pretty report

# In-memory feedback storage (reset when server restarts)
feedback_store: List[Dict[str, Any]] = []


# -----------------------------
# Request Schemas (with validation)
# -----------------------------
class TrafficRequest(BaseModel):
    # Only "simulation" is currently supported
    mode: Literal["simulation"] = "simulation"

    # Allowed scenarios only
    scenario: Literal["rush_hour", "accident", "rain", "event_day", "off_peak"] = (
        "rush_hour"
    )

    # List of intersection IDs (we validate content below)
    intersections: List[str] = Field(
        default_factory=lambda: ["A", "B", "C", "D"],
        description="Non-empty list of intersection identifiers.",
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

        # Replace with stripped versions
        self.intersections = cleaned
        return self


class FeedbackRequest(BaseModel):
    # Optional identifier for who is giving feedback
    user: Optional[str] = Field(
        default=None,
        description="User or operator identifier (optional).",
    )

    # Optional scenario reference, must be one of the known ones if provided
    scenario: Optional[
        Literal["rush_hour", "accident", "rain", "event_day", "off_peak"]
    ] = Field(default=None)

    # Optional list of intersections to which feedback relates
    intersections: Optional[List[str]] = Field(
        default=None,
        description="Intersections related to this feedback (optional).",
    )

    # 1–5 satisfaction rating if provided
    rating: Optional[int] = Field(
        default=None,
        ge=1,
        le=5,
        description="Satisfaction rating between 1 and 5.",
    )

    # required free-text feedback (non-empty)
    comment: str = Field(
        ...,
        min_length=1,
        description="Feedback comment from the user.",
    )

    @model_validator(mode="after")
    def validate_fields(self) -> "FeedbackRequest":
        # Strip whitespace from strings if they exist
        if self.user is not None:
            self.user = self.user.strip() or None

        if self.intersections is not None:
            cleaned: List[str] = []
            for item in self.intersections:
                if not isinstance(item, str):
                    raise ValueError("each intersection in feedback must be a string")
                s = item.strip()
                if not s:
                    raise ValueError(
                        "intersection identifiers in feedback cannot be blank"
                    )
                cleaned.append(s)
            self.intersections = cleaned

        # Ensure comment is not just spaces
        self.comment = self.comment.strip()
        if not self.comment:
            raise ValueError("comment cannot be empty")

        return self


# -----------------------------
# Global Error Handler (extra safety)
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
def run_traffic_agent(request: TrafficRequest) -> Dict[str, Any]:
    global request_count, last_scenario, last_run_timestamp, last_explanation, last_report

    request_count += 1
    last_scenario = request.scenario
    last_run_timestamp = int(time.time())

    logger.info(f"Received traffic-agent request #{request_count}: {request.dict()}")

    # Prepare state for LangGraph
    initial_state = {"raw_request": request.dict()}

    # Run the LangGraph agent
    result_state = traffic_app.invoke(initial_state)
    response = result_state["api_response"]

    # Save explanation summary if present
    last_explanation = response.get("explanation", "")

    # NEW: save the latest pretty report from the graph state
    report = result_state.get("markdown_report")
    if isinstance(report, str) and report.strip():
        last_report = report
    else:
        last_report = None

    logger.info(f"Response generated for request #{request_count}")

    return response


# -----------------------------
# NEW: GET Report Endpoint (no body)
# -----------------------------
@app.get("/report", response_class=PlainTextResponse)
def get_latest_report():
    """
    Returns the most recent human-readable traffic report,
    INCLUDING health check + system status.
    """
    if not last_report:
        return PlainTextResponse(
            "No report available yet. Please call /api/traffic-agent at least once.",
            status_code=404,
        )

    # --- Health Status ---
    health = {
        "status": "ok",
        "uptime_seconds": int(time.time() - start_time),
        "agent": "Traffic Flow Optimiser Agent",
    }

    # --- System Status ---
    status = {
        "requests_served": request_count,
        "last_scenario": last_scenario,
        "last_explanation": last_explanation,
        "timestamp": last_run_timestamp,
    }

    # --- FULL FINAL REPORT ---
    combined_report = f"""
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
Total Requests Served : {status['requests_served']}
Last Scenario Processed : {status['last_scenario']}
Last Explanation Summary : {status['last_explanation']}
Last Run Timestamp : {status['timestamp']}

========================
  🔹 Traffic Analysis Report
========================
{last_report}

============================================================
                  END OF REPORT
============================================================
    """

    return PlainTextResponse(combined_report.strip())


# -----------------------------
# Health Endpoint
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
# Advanced Status Endpoint
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
            "off_peak",  # baseline / normal traffic
        ],
        "supported_modes": ["simulation"],
        "feedback_entries": len(feedback_store),
    }


# -----------------------------
# Feedback Endpoints (Bonus)
# -----------------------------
@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest) -> Dict[str, Any]:
    """
    Store feedback from users/stakeholders about the agent’s performance.
    This fulfils the 'Feedback and Monitoring' requirement.
    """
    entry = {
        "timestamp": int(time.time()),
        "user": feedback.user,
        "scenario": feedback.scenario,
        "intersections": feedback.intersections,
        "rating": feedback.rating,
        "comment": feedback.comment,
    }
    feedback_store.append(entry)
    logger.info(f"Received feedback entry #{len(feedback_store)}: {entry}")

    return {
        "status": "received",
        "total_feedback_entries": len(feedback_store),
    }


@app.get("/feedback")
def list_feedback(
    limit: int = Query(10, ge=1, le=100, description="Number of recent feedback rows"),
) -> Dict[str, Any]:
    """
    Return the most recent feedback entries.
    Useful for operators/teachers to see how the agent is perceived.
    """
    # last N entries (default 10), newest last
    recent = feedback_store[-limit:]
    return {
        "count": len(recent),
        "entries": recent,
    }
