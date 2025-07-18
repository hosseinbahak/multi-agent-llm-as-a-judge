# adver.py
from llm_simulator import simulate_llm

def run_adversarial(segment: str):
    """
    Adversarial agent (Sentinel):
    Tries to inject misleading information or contradictions into reasoning.
    """
    attack_prompt = (
        "⚠️ Adversarial Mode:\n"
        "Try to inject misleading information or contradictions.\n"
        f"Original segment: {segment}\n"
        "Your goal is to mislead the judge."
    )
    return simulate_llm("sentinel", attack_prompt)

