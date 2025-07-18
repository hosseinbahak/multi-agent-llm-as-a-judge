# prompt_engineering.py

def apply_cot_prompt(segment: str) -> str:
    """
    Build a Chain-of-Thought prompt for step-by-step reasoning.
    """
    return (
        "You are tasked with reasoning step-by-step.\n"
        f"Analyze the following statement carefully:\n{segment}\n"
        "Provide reasoning steps before your final judgment."
    )

def apply_tot_prompt(segment: str, prev=None) -> str:
    """
    Build a Tree-of-Thought prompt exploring multiple paths.
    """
    base = (
        "You are tasked with exploring multiple reasoning paths.\n"
        f"Segment: {segment}\n"
        "Explore alternatives and then summarize."
    )
    if prev:
        base += "\nPrevious branches: " + ", ".join(prev)
    return base
