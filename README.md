# Multi-Agent-llm-as-a-judge

This project implements a multi‑agent LLM‑as‑a‑Judge architecture in a modular way.
It is designed to evaluate reasoning tasks by simulating multiple specialized judges, building a reasoning graph, iterating through debate rounds, and finally producing an Explainable Adjudication Report (XAR).
An adversarial agent (Sentinel) is also included to test the robustness of the system against prompt injection or misleading reasoning.

# Project Structure


```
project/
│
├── preprocessing.py          # Normalize and split input text into segments
├── prompt_engineering.py     # Prompt engineering (CoT / ToT) for reasoning
├── llm_simulator.py          # Simulate LLM agents or connect to APIs
├── graph.py                  # Dynamic argument graph data structure
├── debate.py                 # Multi-agent debate & MCTS-like exploration
├── aggregator.py             # Aggregate results & compute scoring
└── pipeline.py               # Main pipeline orchestrating all modules
```

# Main Workflow (pipeline.py)

The file pipeline.py orchestrates the whole process.
Its steps are:
## Preprocessing

Uses preprocessing.py to normalize the raw input text and split it into segments or sentences.
## Prompt Engineering

For each segment, prompt_engineering.py generates a Chain‑of‑Thought (CoT) prompt that encourages step‑by‑step reasoning.
## Initial Judgments

For each prompt, three judge roles are simulated through llm_simulator.py:

    Logician: Detects logical errors and contradictions.

    Innovator: Suggests alternative reasoning paths.

    Synthesizer: Integrates and summarizes multiple viewpoints.

## Adversarial Testing (Sentinel)

The module adver.py implements an adversarial agent (Sentinel) that tries to inject misleading reasoning or attacks into the process.
Its output is also added to the argument graph, allowing you to test how robust the system is.
## Debate Rounds

Through debate.py, the system performs several iterative debate rounds.
Each round expands the Dynamic Argument Graph (graph.py) by letting judge roles critique and build upon previous nodes, similar to an MCTS (Monte Carlo Tree Search) exploration of reasoning paths.
## Aggregation

After debate rounds, aggregator.py collects all outputs from the judges, evaluates them, and computes:

    A final verdict (e.g., “Reasoning sound” or “Flawed reasoning detected”),

    A confidence score.

## Report Generation

pipeline.py builds an Explainable Adjudication Report (XAR) in JSON format.
This includes:

    Summary with verdict and confidence,

    A full dump of the reasoning graph,

    Metadata for logging and potential fine‑tuning.


# Modules Overview
Module	Responsibility
preprocessing.py	Cleans input text and splits into reasoning units.
prompt_engineering.py	Builds CoT/ToT prompts for deeper reasoning.
llm_simulator.py	Simulates LLM responses (can be swapped with real API calls).
adver.py	Implements the adversarial (Sentinel) agent.
graph.py	Maintains a dynamic graph of reasoning and debate.
debate.py	Expands the graph through iterative debate rounds.
aggregator.py	Scores and aggregates judge outputs to reach a verdict.
pipeline.py	Orchestrates all modules into one coherent workflow.

# Features
- Multi‑Agent Judges – Each role (logician, innovator, synthesizer) focuses on a different aspect of reasoning.
- Adversarial Agent – Sentinel actively tests system robustness by trying to mislead judges.
- Dynamic Argument Graph – Every judgment and critique is stored as a node in a graph, showing how reasoning evolves.
- Debate Rounds (MCTS‑like) – Iterative refinement of judgments through multi‑round exploration.
- Modular Design – Each function is in its own file, making it easy to replace the simulator with real LLM APIs or extend features.
- Explainable Output – Generates a JSON report (xar_report.json) for analysis or future model fine‑tuning.