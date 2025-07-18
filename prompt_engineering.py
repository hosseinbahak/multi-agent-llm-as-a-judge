# prompt_engineering.py
def apply_cot_prompt(segment: str) -> str:
    """
    Chain-of-Thought prompt: guide the model to reason step-by-step.
    """
    return (
        "You are tasked with reasoning step-by-step.\n"
        f"Analyze the following statement carefully:\n"
        f"{segment}\n"
        "Provide reasoning steps before your final judgment."
    )

def apply_tot_prompt(segment: str, previous_branches=None) -> str:
    """
    Tree-of-Thought prompt: explore multiple reasoning branches.
    """
    base = (
        "You are tasked with exploring multiple reasoning paths (Tree-of-Thought).\n"
        f"Consider this segment:\n{segment}\n"
        "Think in branches, explore alternatives, then provide evaluations."
    )
    if previous_branches:
        base += "\nAlso consider previous branches: " + ", ".join(previous_branches)
    return base

