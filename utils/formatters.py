# multi_agent_llm_judge/utils/formatters.py
from ..core.data_models import FinalJudgment, RoundResult, Verdict

def format_final_judgment(judgment: FinalJudgment) -> str:
    """Creates a human-readable string summary of the final judgment."""
    
    final_verdict = "CORRECT" if judgment.final_verdict else "INCORRECT"
    output = [
        "==================================================",
        "          ROUND-TABLE JUDGE FINAL REPORT          ",
        "==================================================",
        f"Final Verdict: {final_verdict}",
        f"Initial Confidence (Jury): {judgment.jury_decision.weighted_confidence:.4f}",
        f"Calibrated Confidence: {judgment.calibrated_confidence:.4f}",
        f"Total Rounds: {judgment.num_rounds_executed}",
        f"Total Cost: ${judgment.total_cost:.5f}",
        f"Total Time: {judgment.total_time_ms / 1000:.2f} seconds",
        "--------------------------------------------------",
        "Jury Deliberation Summary:",
        f"  - Consensus Level: {judgment.jury_decision.consensus_level:.2%}",
        f"  - Votes Cast: {len(judgment.jury_decision.votes)}",
        "--------------------------------------------------",
        "Round-by-Round Analysis:",
    ]
    
    for i, round_result in enumerate(judgment.round_results):
        output.append(f"
[ Round {i+1} ]")
        if not round_result.agent_analyses:
            output.append("  No agent analyses in this round.")
            continue
            
        for analysis in round_result.agent_analyses:
            verdict_symbol = "✅" if analysis.verdict == Verdict.CORRECT else "❌" if analysis.verdict == Verdict.INCORRECT else "❔"
            output.append(
                f"  - {analysis.agent_name:<20} | {verdict_symbol} {analysis.verdict.name:<9} | Confidence: {analysis.confidence:.2f}"
            )
            output.append(f"    Summary: {analysis.analysis[:150]}...")

    output.append("==================================================")
    return "
".join(output)

