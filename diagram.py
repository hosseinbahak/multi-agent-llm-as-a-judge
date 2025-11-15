import graphviz

def create_code_flow_diagram():
    """
    Generates a sequence diagram for the multi-agent LLM judge system.
    """
    # Initialize a new directed graph
    dot = graphviz.Digraph('Multi_Agent_LLM_Judge_Flow', comment='Code Flow Diagram')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.6', ranksep='1.2')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Helvetica')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # Define main components (lifelines)
    with dot.subgraph(name='cluster_main_components') as c:
        c.attr(label='Core Components', style='rounded', color='gray')
        c.node('main', 'main.py\n(Entry Point)', shape='ellipse', fillcolor='lightgreen')
        c.node('round_manager', 'RoundManager', fillcolor='orange')
        c.node('agent_manager', 'AgentManager')
        c.node('jury_manager', 'JuryManager')
        c.node('model_manager', 'ModelManager')
        c.node('cache_manager', 'CacheManager')
        c.node('collector', 'CalibrationDataCollector', fillcolor='lightgoldenrodyellow')

    # Define agents and external systems
    with dot.subgraph(name='cluster_agents') as c:
        c.attr(label='Specialized Agents', style='rounded', color='gray')
        c.node('agent_n', 'Agent 1...N\n(e.g., CoT, Adversary)', shape='component', fillcolor='whitesmoke')

    with dot.subgraph(name='cluster_external') as c:
        c.attr(label='External Systems', style='rounded', color='gray')
        c.node('llm_provider', 'LLM Provider\n(e.g., OpenRouter)', shape='cylinder', fillcolor='pink')
        c.node('user', 'User/Test Runner', shape='actor', fillcolor='none')
        c.node('filesystem', 'File System\n(Cache, Data)', shape='folder', fillcolor='tan')

    # --- Sequence of Events ---

    # 1. Initialization
    dot.edge('user', 'main', label='1. Run script')
    dot.edge('main', 'model_manager', label='2. Initialize Provider')
    dot.edge('main', 'cache_manager', label='3. Initialize Cache')
    dot.edge('main', 'round_manager', label='4. Initialize RoundManager')
    dot.edge('round_manager', 'agent_manager', label='5. Initialize AgentManager')
    dot.edge('agent_manager', 'agent_n', label='6. Discover/Register Agents')
    dot.edge('main', 'collector', label='7. Initialize DataCollector')

    # 2. Evaluation Request
    dot.edge('main', 'round_manager', label='8. evaluate(request)', style='bold', color='blue', fontcolor='blue')

    # 3. Inside RoundManager.evaluate
    dot.edge('round_manager', 'cache_manager', label='9. Check cache for result')
    dot.edge('round_manager', 'agent_manager', label='10. create_agents()', penwidth='2')
    dot.edge('round_manager', 'round_manager', label='11. Loop: Execute Rounds\n(for round_num in num_rounds)', shape='loop')

    # 4. Inside a Round
    with dot.subgraph(name='cluster_round') as c:
        c.attr(label='Inside a Single Round', style='dashed', color='blue')
        c.edge('round_manager', 'agent_manager', label='12. execute_agents_parallel()', color='darkgreen')
        c.edge('agent_manager', 'agent_n', label='13. analyze()', color='darkgreen')
        c.edge('agent_n', 'model_manager', label='14. Request LLM Call', color='darkgreen')
        c.edge('model_manager', 'llm_provider', label='15. API Call', color='purple')
        c.edge('llm_provider', 'model_manager', label='16. LLM Response', color='purple')
        c.edge('model_manager', 'agent_n', label='17. Return result', color='darkgreen')
        c.edge('agent_n', 'agent_manager', label='18. Return AgentAnalysis', color='darkgreen')
        c.edge('agent_manager', 'round_manager', label='19. Return List[AgentAnalysis]', color='darkgreen')
        c.edge('round_manager', 'round_manager', label='20. Create RoundSummary', style='dotted')

    # 5. Jury Deliberation
    dot.edge('round_manager', 'jury_manager', label='21. conduct_trial(aggregated_analyses)', style='bold', color='firebrick')
    dot.edge('jury_manager', 'model_manager', label='22. Request LLM Call for Deliberation', color='firebrick')
    dot.edge('model_manager', 'llm_provider', label='23. API Call', color='purple')
    dot.edge('llm_provider', 'model_manager', label='24. LLM Response', color='purple')
    dot.edge('model_manager', 'jury_manager', label='25. Return deliberation result', color='firebrick')
    dot.edge('jury_manager', 'round_manager', label='26. Return JuryDecision', style='bold', color='firebrick')

    # 6. Finalization and Output
    dot.edge('round_manager', 'round_manager', label='27. Create FinalJudgment', style='dotted')
    dot.edge('round_manager', 'main', label='28. Return EvaluationResult', style='bold', color='blue', fontcolor='blue')
    dot.edge('main', 'collector', label='29. save_unlabeled_evaluation(result)')
    dot.edge('collector', 'filesystem', label='30. Write .json & .pkl files')
    dot.edge('main', 'collector', label='31. create_labeling_batch()')
    dot.edge('collector', 'filesystem', label='32. Write labeling_batch.json')
    dot.edge('main', 'user', label='33. Print summary')

    # Render the graph
    output_filename = 'multi_agent_system_flow'
    try:
        dot.render(output_filename, format='png', view=False, cleanup=True)
        print(f"✅ Diagram saved successfully as '{output_filename}.png'")
    except graphviz.backend.ExecutableNotFound:
        print("\n❌ Error: Graphviz executable not found.")
        print("Please install Graphviz on your system and ensure it's in your PATH.")
        print(" - macOS (Homebrew): brew install graphviz")
        print(" - Ubuntu/Debian: sudo apt-get install graphviz")
        print(" - Windows: Download from https://graphviz.org/download/ and add to PATH.")
        # Save the DOT source file anyway
        dot.save(f'{output_filename}.dot')
        print(f"ℹ️ DOT source file saved as '{output_filename}.dot'. You can render it manually.")


if __name__ == '__main__':
    create_code_flow_diagram()

