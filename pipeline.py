# pipeline.py
import json
from preprocessing import normalize_text, split_into_segments
from prompt_engineering import apply_cot_prompt
from llm_simulator import simulate_llm
from graph import ArgumentGraph
from debate import debate_round
from aggregator import aggregate_results

def judge_pipeline(raw_text: str, rounds: int = 2):
    # Pre-processing
    normalized = normalize_text(raw_text)
    segments = split_into_segments(normalized)

    graph = ArgumentGraph()
    root = graph.add_node("input", raw_text, 1.0)

    # Initial judgments for each segment
    for seg in segments:
        prompt = apply_cot_prompt(seg)
        for role in ["logician", "innovator", "synthesizer"]:
            txt, conf = simulate_llm(role, prompt)
            nid = graph.add_node(role, txt, conf)
            graph.add_edge(root, nid)

        # Sentinel test
        s_txt, s_conf = simulate_llm("sentinel", prompt)
        sid = graph.add_node("sentinel", s_txt, s_conf)
        graph.add_edge(root, sid)

    # Debate rounds
    for r in range(rounds):
        debate_round(graph, round_num=r+1)

    # Aggregation
    verdict, conf = aggregate_results(graph)

    xar_report = {
        "summary": {
            "final_verdict": verdict,
            "confidence": conf,
            "security_warning": "Sentinel actively tested bias injection."
        },
        "argument_graph": graph.to_dict(),
        "rounds_executed": rounds,
        "data_logging": {
            "stored_for_future_fine_tuning": True,
            "total_nodes": len(graph.nodes)
        }
    }

    return xar_report

if __name__ == "__main__":
    input_text = "این استدلال شامل یک تناقض منطقی است. جمله‌ی دوم هم مشکل دارد."
    report = judge_pipeline(input_text, rounds=2)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    with open("xar_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("✅ XAR report saved.")
