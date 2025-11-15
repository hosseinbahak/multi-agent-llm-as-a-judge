# multi_agent_llm_judge/utils/round_formatter.py
from typing import List
from ..core.data_models import RoundSummary, AgentAnalysis, Verdict
import json

def format_round_as_json(round_summary: RoundSummary) -> str:
    """Format round data as pretty JSON."""
    data = {
        "round": round_summary.round_number,
        "metrics": {
            "average_confidence": round_summary.average_confidence,
            "confidence_variance": round_summary.confidence_variance,
            "total_cost": round_summary.total_cost,
            "processing_time_ms": round_summary.processing_time_ms
        },
        "agents": []
    }
    
    for analysis in round_summary.agent_analyses:
        agent_data = {
            "name": analysis.agent_name,
            "type": analysis.agent_type,
            "model": analysis.model_used,
            "verdict": analysis.verdict.value if analysis.verdict else None,
            "confidence": analysis.confidence,
            "analysis_preview": analysis.analysis[:300] + "...",
            "reasoning_steps": analysis.reasoning_steps[:3],
            "evidence_count": len(analysis.evidence),
            "assumptions": analysis.assumptions[:3]
        }
        data["agents"].append(agent_data)
    
    data["consensus"] = round_summary.consensus_points[:5]
    data["disagreements"] = round_summary.disagreement_points[:5]
    
    return json.dumps(data, indent=2, ensure_ascii=False)

def create_round_comparison_table(rounds: List[RoundSummary]) -> str:
    """Create a comparison table across rounds."""
    lines = []
    lines.append("Round Comparison Table")
    lines.append("=" * 100)
    lines.append(f"{'Round':<8} {'Agents':<8} {'Avg Conf':<12} {'Variance':<12} {'Cost':<10} {'Time (ms)':<12}")
    lines.append("-" * 100)
    
    for r in rounds:
        lines.append(
            f"{r.round_number:<8} "
            f"{len(r.agent_analyses):<8} "
            f"{r.average_confidence:<12.2%} "
            f"{r.confidence_variance:<12.4f} "
            f"${r.total_cost:<9.4f} "
            f"{r.processing_time_ms:<12}"
        )
    
    return "\n".join(lines)
