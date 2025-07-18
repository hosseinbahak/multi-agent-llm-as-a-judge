# aggregator.py
def aggregate_results(graph):
    """
    Aggregate results from judges using weighted confidence scores.
    """
    logician_scores = []
    overall_conf = []

    for node in graph.nodes.values():
        if node["role"] in ["logician", "innovator", "synthesizer"]:
            overall_conf.append(node["confidence"])
            if node["role"] == "logician":
                logician_scores.append(("❌" in node["text"], node["confidence"]))

    # Weighted contradiction detection
    total_conf = sum(c for _, c in logician_scores)
    contradiction_conf = sum(c for is_err, c in logician_scores if is_err)
    flawed_ratio = (contradiction_conf / total_conf) if total_conf > 0 else 0

    verdict = "❌ Flawed reasoning detected" if flawed_ratio > 0.4 else "✅ Reasoning seems sound"
    confidence = round(sum(overall_conf) / max(len(overall_conf), 1), 2)

    return verdict, confidence
