from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
import random


# 1. Define the state that flows between nodes
class TrafficState(TypedDict, total=False):
    raw_request: Dict[str, Any]
    traffic_data: List[Dict[str, Any]]
    congestion: Dict[str, Any]
    signal_plan: Dict[str, Any]
    metrics: Dict[str, Any]
    api_response: Dict[str, Any]
    scenario: str
    explanation: str
    # New bonus fields
    predicted_congestion: Dict[str, Any]
    weather_severity: Dict[str, Any]
    incident_probability: float
    recommendations: List[str]
    # New: pretty text report (not returned in JSON)
    markdown_report: str


# 2. Helper functions (synthetic but slightly more realistic logic)

def simulate_traffic(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create synthetic traffic data for a scenario.

    Supports scenarios:
    - "rush_hour"   : high volume, normal speed
    - "accident"    : very high volume, low speed
    - "rain"        : high volume, reduced speed, longer queues
    - "event_day"   : very high volume, long queues
    - anything else : normal / baseline traffic
    """
    scenario = params.get("scenario", "rush_hour").lower()

    if scenario == "rush_hour":
        base_volume = 80
        base_speed = 25
    elif scenario == "accident":
        base_volume = 95
        base_speed = 15
    elif scenario == "rain":
        base_volume = 85
        base_speed = 18
    elif scenario == "event_day":
        base_volume = 100
        base_speed = 22
    else:
        # default / off-peak
        base_volume = 60
        base_speed = 30

    intersections = params.get("intersections", ["A", "B", "C", "D"])

    data = []
    for name in intersections:
        # Add a little randomness so every intersection is not identical
        volume = base_volume + random.randint(-3, 3)
        queue_length = max(0, volume // 2 + random.randint(-2, 2))
        avg_speed = max(5, base_speed + random.randint(-3, 3))

        data.append({
            "intersection": name,
            "volume": volume,         # vehicles per cycle
            "queue_length": queue_length,
            "avg_speed": avg_speed,   # km/h
        })
    return data


def analyse_congestion(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple congestion detection based on volume and queue length.
    """
    congestion_levels = {}
    hotspots = []

    for d in data:
        vol = d["volume"]
        if vol > 90:
            level = "severe"
        elif vol > 70:
            level = "high"
        elif vol > 50:
            level = "medium"
        else:
            level = "low"

        name = d["intersection"]
        congestion_levels[name] = {
            "volume": vol,
            "queue_length": d["queue_length"],
            "avg_speed": d.get("avg_speed", 0),
            "level": level,
        }

        if level in ("high", "severe"):
            hotspots.append(name)

    return {
        "levels": congestion_levels,
        "hotspots": hotspots,
    }


def optimise_signals(congestion: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adaptive signal timing:
    - more green time for higher congestion
    - keeps total cycle around 90 seconds
    """
    levels = congestion.get("levels", {})
    plan = {}

    for name, info in levels.items():
        level = info["level"]

        if level == "severe":
            green = 70
        elif level == "high":
            green = 60
        elif level == "medium":
            green = 45
        else:
            green = 30

        plan[name] = {
            "green_seconds": green,
            "red_seconds": 90 - green,
        }

    return plan


def calculate_metrics(data: List[Dict[str, Any]],
                      signal_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dummy performance metrics with a bit more detail.
    """
    if not data:
        return {
            "total_volume": 0,
            "average_queue_length": 0,
            "num_intersections": 0,
            "average_speed": 0,
            "max_queue_length": 0,
        }

    total_volume = sum(d["volume"] for d in data)
    avg_queue = sum(d["queue_length"] for d in data) / len(data)
    avg_speed = sum(d.get("avg_speed", 0) for d in data) / len(data)
    max_queue = max(d["queue_length"] for d in data)

    return {
        "total_volume": total_volume,
        "average_queue_length": avg_queue,
        "num_intersections": len(data),
        "average_speed": avg_speed,
        "max_queue_length": max_queue,
    }


# 3. Node functions for LangGraph

def parse_input(state: TrafficState) -> TrafficState:
    raw = state["raw_request"]
    # Store scenario separately for explanation & status
    state["scenario"] = raw.get("scenario", "rush_hour")
    return state


def load_or_simulate_traffic(state: TrafficState) -> TrafficState:
    raw = state["raw_request"]
    mode = raw.get("mode", "simulation")

    if mode == "simulation":
        state["traffic_data"] = simulate_traffic(raw)
    else:
        # For now we only support simulation.
        # You can add real API calls here later without changing other nodes.
        state["traffic_data"] = simulate_traffic(raw)

    return state


def detect_congestion(state: TrafficState) -> TrafficState:
    data = state["traffic_data"]
    state["congestion"] = analyse_congestion(data)
    return state


def compute_signal_plan(state: TrafficState) -> TrafficState:
    congestion = state["congestion"]
    state["signal_plan"] = optimise_signals(congestion)
    return state


def compute_metrics(state: TrafficState) -> TrafficState:
    data = state["traffic_data"]
    plan = state["signal_plan"]
    state["metrics"] = calculate_metrics(data, plan)
    return state


def compute_predictions(state: TrafficState) -> TrafficState:
    """
    Bonus node:
    - Predicts congestion trend per intersection.
    - Assigns scenario/weather severity.
    - Estimates incident probability.
    - Generates smart recommendations.
    """
    scenario = state.get("scenario", "rush_hour").lower()
    data = state.get("traffic_data", [])
    congestion = state.get("congestion", {})
    metrics = state.get("metrics", {})

    levels = congestion.get("levels", {})
    hotspots = congestion.get("hotspots", [])
    predicted: Dict[str, Any] = {}

    # Predict congestion trend per intersection
    for d in data:
        name = d["intersection"]
        info = levels.get(name, {})
        level = info.get("level", "low")
        queue = d.get("queue_length", 0)
        speed = d.get("avg_speed", 0)

        if queue > 45 or (level in ("high", "severe") and speed < 20):
            trend = "likely_increase"
        elif level in ("medium", "high"):
            trend = "stable"
        else:
            trend = "likely_decrease"

        predicted[name] = {
            "current_level": level,
            "trend": trend,
        }

    # Scenario / weather severity
    weather_detail: Dict[str, Any] = {}
    if scenario == "rain":
        weather_detail["type"] = "rain"
        weather_detail["severity"] = random.choice(["light", "moderate", "heavy"])
    elif scenario == "event_day":
        weather_detail["type"] = "event_day"
        weather_detail["severity"] = random.choice(["local_event", "major_event"])
    elif scenario == "accident":
        weather_detail["type"] = "accident"
        weather_detail["severity"] = random.choice(["minor", "major"])
    else:
        weather_detail["type"] = scenario
        weather_detail["severity"] = "normal"

    # Determine overall congestion level (worst intersection)
    level_priority = {"low": 1, "medium": 2, "high": 3, "severe": 4}
    worst_level = "low"
    for info in levels.values():
        lvl = info.get("level", "low")
        if level_priority.get(lvl, 1) > level_priority.get(worst_level, 1):
            worst_level = lvl

    # Incident probability (very rough, but looks smart)
    base_probs = {"low": 0.05, "medium": 0.15, "high": 0.30, "severe": 0.50}
    base = base_probs.get(worst_level, 0.1)
    prob = base + random.uniform(-0.03, 0.05)  # small noise
    prob = max(0.0, min(prob, 0.95))           # clamp between 0 and 0.95

    # Smart recommendations
    recommendations: List[str] = []
    if hotspots:
        recommendations.append(
            f"Increase green time at congestion hotspots: {', '.join(hotspots)} in the next few cycles."
        )
    if worst_level in ("high", "severe"):
        recommendations.append(
            "Deploy traffic wardens or police near the busiest intersections to manage queues."
        )
    if scenario in ("rain", "event_day"):
        recommendations.append(
            "Inform drivers about expected delays via roadside message boards or mobile notifications."
        )
    if not recommendations:
        recommendations.append(
            "Current timing plan is sufficient; continue monitoring queues and speeds."
        )

    # Store in state
    state["predicted_congestion"] = predicted
    state["weather_severity"] = weather_detail
    state["incident_probability"] = round(prob, 2)
    state["recommendations"] = recommendations

    return state


def generate_explanation(state: TrafficState) -> TrafficState:
    """
    Create a human-readable explanation of what the agent did.
    This makes the agent look more 'intelligent' and is great in demos.
    """
    scenario = state.get("scenario", "rush_hour")
    congestion = state.get("congestion", {})
    metrics = state.get("metrics", {})

    levels = congestion.get("levels", {})
    hotspots = congestion.get("hotspots", [])

    # Determine overall congestion level (worst intersection)
    level_priority = {"low": 1, "medium": 2, "high": 3, "severe": 4}
    worst_level = "low"
    for info in levels.values():
        lvl = info.get("level", "low")
        if level_priority.get(lvl, 1) > level_priority.get(worst_level, 1):
            worst_level = lvl

    total_volume = metrics.get("total_volume", 0)
    avg_queue = round(metrics.get("average_queue_length", 0), 2)
    num_intersections = metrics.get("num_intersections", 0)

    if hotspots:
        hotspot_str = ", ".join(hotspots)
    else:
        hotspot_str = "none"

    explanation = (
        f"Scenario '{scenario}' resulted in overall {worst_level} congestion "
        f"across {num_intersections} intersections with total volume {total_volume} "
        f"vehicles and an average queue length of {avg_queue} vehicles. "
        f"Identified congestion hotspots at: {hotspot_str}. "
        f"Signal timings were adjusted to prioritise these busy intersections."
    )

    # Add predictive + incident info
    predicted = state.get("predicted_congestion", {})
    worsening = [name for name, info in predicted.items()
                 if info.get("trend") == "likely_increase"]
    if worsening:
        explanation += (
            f" Predicted congestion is likely to increase soon at: {', '.join(worsening)}."
        )
    else:
        explanation += (
            " Predicted congestion is stable or improving across all intersections."
        )

    incident_prob = state.get("incident_probability")
    if incident_prob is not None:
        explanation += (
            f" Estimated incident escalation probability is around {int(incident_prob * 100)}%."
        )

    state["explanation"] = explanation
    return state


def build_markdown_report(state: TrafficState) -> TrafficState:
    """
    Build a PRETTY PLAIN-TEXT report (no markdown symbols).
    Stored in state['markdown_report'], but not returned in JSON.
    """
    scenario = state.get("scenario", "rush_hour")
    metrics = state.get("metrics", {})
    congestion = state.get("congestion", {})
    levels = congestion.get("levels", {})
    hotspots = congestion.get("hotspots", [])
    predictions = state.get("predicted_congestion", {})
    weather = state.get("weather_severity", {})
    incident_prob = state.get("incident_probability", 0.0)
    recommendations = state.get("recommendations", [])
    explanation = state.get("explanation", "")

    num_intersections = metrics.get("num_intersections", 0)
    hot_str = ", ".join(hotspots) if hotspots else "none"

    # Build per-intersection text blocks
    intersections_block_lines: List[str] = []
    for name, info in levels.items():
        intersections_block_lines.append(
            f"  - {name}:\n"
            f"      Level      : {info.get('level','')}\n"
            f"      Volume     : {info.get('volume',0)} vehicles\n"
            f"      Queue      : {info.get('queue_length',0)} vehicles\n"
            f"      Avg speed  : {info.get('avg_speed',0)} km/h"
        )
    intersections_block = "\n".join(intersections_block_lines) if intersections_block_lines else "  (no intersections)\n"

    # Prediction text
    prediction_lines: List[str] = []
    for name, info in predictions.items():
        prediction_lines.append(
            f"  - {name}: {info.get('current_level','')} congestion, trend: {info.get('trend','')}"
        )
    prediction_block = "\n".join(prediction_lines) if prediction_lines else "  (no prediction data)\n"

    # Recommendations text
    if recommendations:
        recommendations_block = "\n".join(f"  - {rec}" for rec in recommendations)
    else:
        recommendations_block = "  - No special actions recommended. Continue monitoring.\n"

    report = f"""
============================================================
              TRAFFIC FLOW OPTIMISER AGENT REPORT
============================================================

1) Scenario Overview
   ------------------
   Scenario                : {scenario}
   Context / Weather       : {weather.get('type','unknown')} ({weather.get('severity','unknown')})
   Estimated Incident Risk : {int(incident_prob * 100)} %

2) Executive Summary
   ------------------
   {explanation}

3) Network Metrics
   ----------------
   Intersections analysed  : {num_intersections}
   Total traffic volume    : {metrics.get('total_volume', 0)} vehicles
   Average queue length    : {metrics.get('average_queue_length', 0):.2f} vehicles
   Average speed           : {metrics.get('average_speed', 0):.1f} km/h
   Maximum queue length    : {metrics.get('max_queue_length', 0)} vehicles
   Congestion hotspots     : {hot_str}

4) Congestion by Intersection
   ---------------------------
{intersections_block}

5) Signal Timing Plan
   -------------------
""".rstrip("\n")

    # Add signal plan section
    signal_plan = state.get("signal_plan", {})
    if signal_plan:
        for name, plan in signal_plan.items():
            report += (
                f"\n  - {name}: green {plan.get('green_seconds',0)} s, "
                f"red {plan.get('red_seconds',0)} s"
            )
    else:
        report += "\n  (no signal plan data)"

    # Add predictions
    report += f"""

6) Predicted Congestion Trend
   ---------------------------
{prediction_block}

7) Recommended Actions
   --------------------
{recommendations_block}

8) How to Read This Report
   ------------------------
   - Use the scenario overview to understand why traffic is behaving this way.
   - Use the network metrics to compare different days or scenarios.
   - Check the 'Congestion by Intersection' section to prioritise which junctions
     need attention first.
   - Apply the suggested signal timings and recommended actions to reduce queues
     and smoothen overall flow.
"""

    state["markdown_report"] = report
    return state


def build_api_response(state: TrafficState) -> TrafficState:
    """
    This is the unified JSON you will return.
    Report string is NOT included here.
    """
    state["api_response"] = {
        "agent_name": "Traffic Flow Optimiser Agent",
        "scenario": state.get("scenario", "rush_hour"),
        "signal_plan": state.get("signal_plan", {}),
        "congestion": state.get("congestion", {}),
        "metrics": state.get("metrics", {}),
        "predicted_congestion": state.get("predicted_congestion", {}),
        "weather_severity": state.get("weather_severity", {}),
        "incident_probability": state.get("incident_probability", 0.0),
        "recommendations": state.get("recommendations", []),
        "explanation": state.get("explanation", ""),
        "status": "ok",
        # NOTE: no report here on purpose
    }
    return state


# 4. Build and compile the graph

def build_graph():
    graph = StateGraph(TrafficState)

    graph.add_node("parse_input", parse_input)
    graph.add_node("load_or_simulate_traffic", load_or_simulate_traffic)
    graph.add_node("detect_congestion", detect_congestion)
    graph.add_node("compute_signal_plan", compute_signal_plan)
    graph.add_node("compute_metrics", compute_metrics)
    graph.add_node("compute_predictions", compute_predictions)
    graph.add_node("generate_explanation", generate_explanation)
    graph.add_node("build_markdown_report", build_markdown_report)
    graph.add_node("build_api_response", build_api_response)

    graph.set_entry_point("parse_input")
    graph.add_edge("parse_input", "load_or_simulate_traffic")
    graph.add_edge("load_or_simulate_traffic", "detect_congestion")
    graph.add_edge("detect_congestion", "compute_signal_plan")
    graph.add_edge("compute_signal_plan", "compute_metrics")
    graph.add_edge("compute_metrics", "compute_predictions")
    graph.add_edge("compute_predictions", "generate_explanation")
    graph.add_edge("generate_explanation", "build_markdown_report")
    graph.add_edge("build_markdown_report", "build_api_response")
    graph.add_edge("build_api_response", END)

    return graph.compile()


# This is the LangGraph app you will import in the API file
traffic_app = build_graph()
