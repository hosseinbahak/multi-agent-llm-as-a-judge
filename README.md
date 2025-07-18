# multi-agent-llm-as-a-judge

A modular architecture implementing a Hybrid Multi‑Agent LLM‑as‑a‑Judge system.
This project simulates or orchestrates multiple specialized judge agents, a sentinel adversarial agent, and a dynamic reasoning graph with debate rounds (MCTS‑like), then produces an explainable adjudication report (XAR).

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

# Features


- ✅ **Preprocessing:** Clean and segment input text.  
- ✅ **Prompt Engineering:** Generate Chain‑of‑Thought and Tree‑of‑Thought prompts.  
- ✅ **Multi‑Agent Judges:** Logician, Innovator, Synthesizer roles simulated or connected to real LLM APIs.  
- ✅ **Sentinel Agent:** Adversarial testing to probe vulnerabilities.  
- ✅ **Dynamic Argument Graph:** Build and traverse a reasoning graph with MCTS‑like debate rounds.  
- ✅ **Aggregation:** Weighted scoring and final verdict.  
- ✅ **Explainable Adjudication Report:** JSON output for analysis and future fine‑tuning.  
