# llm_simulator.py
import random

PROMPTS = {
    "logician": "You are The Logician. Focus on logical consistency and contradictions.",
    "innovator": "You are The Innovator. Suggest alternative reasoning paths.",
    "synthesizer": "You are The Synthesizer. Summarize and integrate viewpoints.",
    "sentinel": "You are The Sentinel. Act adversarial and test with biases."
}

def simulate_llm(role: str, text: str):
    """
    Simulate an LLM response for a given role with a confidence score.
    """
    base = PROMPTS.get(role, "You are a helpful evaluator.")
    conf = round(random.uniform(0.6, 1.0), 2)

    if role == "logician":
        if "تناقض" in text or "contradiction" in text.lower():
            return f"{base}\n❌ Logical contradiction detected.", conf
        else:
            return f"{base}\n✅ No major logical flaw detected.", conf

    if role == "innovator":
        return f"{base}\n💡 Alternative reasoning path suggested.", conf

    if role == "synthesizer":
        return f"{base}\n📌 Summarized and integrated viewpoints.", conf

    if role == "sentinel":
        return f"{base}\n⚠️ Injected adversarial bias and tested responses.", conf

    return f"{base}\n[No specific output]", conf
