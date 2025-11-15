# -*- coding: utf-8 -*-
"""
MetaQAAgent — evaluates the QA process: instruction-following, evidence sufficiency vs claimed certainty, clarity.
"""

from typing import List, Optional
from ..base_agent import BaseAgent
from ...core.data_models import AgentAnalysis, AgentType, Evidence, Verdict

class MetaQAAgent(BaseAgent):
    """Agent that evaluates the QA process quality."""

    agent_type = AgentType.META_QA
    default_system_prompt = """You evaluate the QA PROCESS itself.

Check:
- Instruction-following: Did the answer obey all constraints (format, scope, safety)?
- Evidence sufficiency vs claimed certainty; warn against treating self-explanations as proof.
- Clarity/directness and coverage of sub-questions.

Output:
- Process strengths/weaknesses
- Missing constraints or scope violations
- Issues: [instruction_misalignment, parametric_knowledge_error]
- Final verdict: CORRECT/INCORRECT/UNCERTAIN
- Confidence level (0-100%)"""

    def _build_prompt(
        self,
        question: str,
        answer: str,
        context: Optional[str],
        previous_analyses: Optional[List[AgentAnalysis]],
        round_number: int
    ) -> str:
        prompt = f"""Question: {question}

Answer to evaluate (process-wise): {answer}

"""
        if context:
            prompt += f"Context: {context}\n\n"

        if previous_analyses:
            prompt += "Previous process evaluations:\n"
            for analysis in previous_analyses[:2]:  # Include max 2 previous
                prompt += f"- {analysis.agent_name}: {analysis.verdict.value}\n"

        prompt += """Evaluate the QA process quality following the specified criteria."""

        return prompt

    def _parse_response(
        self,
        response_text: str,
        model_id: str,
        tokens_used: int,
        cost: float,
        processing_time_ms: int
    ) -> AgentAnalysis:
        """Parse the meta QA evaluation."""
        # Extract verdict
        verdict_bool = self._extract_verdict(response_text)
        
        # Process quality issues can affect verdict
        if self._has_severe_process_issues(response_text):
            verdict = Verdict.UNCERTAIN
        elif verdict_bool is True:
            verdict = Verdict.CORRECT
        elif verdict_bool is False:
            verdict = Verdict.INCORRECT
        else:
            verdict = Verdict.UNCERTAIN

        # Extract confidence
        confidence = self._extract_confidence(response_text)
        
        # Extract process evaluation evidence
        evidence = self._extract_process_evidence(response_text)
        
        # Extract strengths and weaknesses
        reasoning_steps = self._extract_process_assessment(response_text)
        
        # Extract process issues
        limitations = self._extract_process_issues(response_text)

        return AgentAnalysis(
            agent_type=self.agent_type,
            agent_name=self.name,
            model_used=model_id,
            analysis=response_text,
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            reasoning_steps=reasoning_steps,
            limitations=limitations,
            tokens_used=tokens_used,
            cost=cost,
            processing_time_ms=processing_time_ms
        )

    def _has_severe_process_issues(self, response: str) -> bool:
        """Check if there are severe process quality issues."""
        severe_keywords = [
            'instruction_misalignment', 'violated constraint', 'ignored format',
            'scope violation', 'failed to follow', 'self-explanation as proof'
        ]
        
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in severe_keywords)

    def _extract_process_evidence(self, response: str) -> List[Evidence]:
        """Extract evidence about process quality."""
        evidence_list = []
        
        # Evidence categories
        evidence_patterns = {
            'instruction_following': ['instruction', 'constraint', 'format', 'scope'],
            'evidence_sufficiency': ['evidence', 'proof', 'certainty', 'self-explanation'],
            'clarity': ['clarity', 'direct', 'coverage', 'sub-question']
        }
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            for category, keywords in evidence_patterns.items():
                if any(keyword in line_lower for keyword in keywords):
                    evidence_list.append(Evidence(
                        source=f"process_{category}",
                        content=line.strip(),
                        relevance_score=0.8
                    ))
                    break
        
        return evidence_list

    def _extract_process_assessment(self, response: str) -> List[str]:
        """Extract process strengths and weaknesses."""
        assessments = []
        
        # Look for strengths/weaknesses sections
        import re
        for section_type in ['strength', 'weakness']:
            pattern = f'{section_type}[s]?:?(.*?)(?:weakness|strength|issues|final|$)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            
            if match:
                section_text = match.group(1)
                # Extract bullet points or numbered items
                items = re.findall(r'[-•*]\s*(.+)|(\d+)[.)\s]+(.+)', section_text)
                for item in items:
                    item_text = ' '.join(part for part in item if part).strip()
                    if len(item_text) > 10:
                        assessments.append(f"{section_type.capitalize()}: {item_text}")
        
        return assessments[:6]

    def _extract_process_issues(self, response: str) -> List[str]:
        """Extract process quality issues."""
        issues = []
        
        # Primary issue types
        issue_types = [
            'instruction_misalignment', 'parametric_knowledge_error',
            'format violation', 'scope violation', 'missing constraint',
            'overclaimed certainty', 'insufficient evidence'
        ]
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            # Check for issues: format
            if 'issues:' in line_lower:
                issues.append(line.strip())
            else:
                # Check for individual issue mentions
                for issue_type in issue_types:
                    if issue_type.replace('_', ' ') in line_lower:
                        issues.append(f"Process issue: {line.strip()}")
                        break
        
        return issues[:5]
