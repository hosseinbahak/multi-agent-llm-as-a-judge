# -*- coding: utf-8 -*-
"""
BiasAuditorAgent — audits confirmation/selection/cultural + position/length bias; suggests mitigations.
"""

from typing import List, Optional
from ..base_agent import BaseAgent
from ...core.data_models import AgentAnalysis, AgentType, Evidence, Verdict

class BiasAuditorAgent(BaseAgent):
    """Agent that audits multiple types of bias."""

    agent_type = AgentType.BIAS_AUDITOR
    default_system_prompt = """You audit bias.

Check for:
- confirmation, selection, cultural/demographic, loaded/gendered language
- POSITION and LENGTH bias: would judgment change if option order or answer length changed? Explain.

Output:
- Bias findings (each → example → why problematic)
- Position/Length bias risk: LOW/MEDIUM/HIGH (+reason)
- Issues: [bias:{position|length|confirmation|selection|cultural}]
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

Answer to audit: {answer}

"""
        if context:
            prompt += f"Context: {context}\n\n"

        prompt += """Audit for multiple bias types including position/length bias."""

        return prompt

    def _parse_response(
        self,
        response_text: str,
        model_id: str,
        tokens_used: int,
        cost: float,
        processing_time_ms: int
    ) -> AgentAnalysis:
        """Parse the bias audit response."""
        # Extract verdict
        verdict_bool = self._extract_verdict(response_text)
        
        # Bias findings affect verdict confidence
        bias_severity = self._assess_bias_severity(response_text)
        if bias_severity == "HIGH":
            verdict = Verdict.UNCERTAIN
        elif verdict_bool is True:
            verdict = Verdict.CORRECT
        elif verdict_bool is False:
            verdict = Verdict.INCORRECT
        else:
            verdict = Verdict.UNCERTAIN

        # Extract confidence
        confidence = self._extract_confidence(response_text)
        
        # Extract bias findings as evidence
        evidence = self._extract_bias_findings(response_text)
        
        # Extract position/length assessment
        reasoning_steps = self._extract_position_length_analysis(response_text)
        
        # Extract bias issues
        limitations = self._extract_bias_issues(response_text)

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

    def _assess_bias_severity(self, response: str) -> str:
        """Assess overall bias severity."""
        response_lower = response.lower()
        
        if 'high' in response_lower and 'bias' in response_lower:
            return "HIGH"
        elif 'medium' in response_lower and 'bias' in response_lower:
            return "MEDIUM"
        else:
            return "LOW"

    def _extract_bias_findings(self, response: str) -> List[Evidence]:
        """Extract specific bias findings with examples."""
        evidence_list = []
        
        bias_types = [
            'confirmation bias', 'selection bias', 'cultural bias',
            'demographic bias', 'position bias', 'length bias',
            'loaded language', 'gendered language'
        ]
        
        import re
        # Look for bias findings with examples
        for bias_type in bias_types:
            pattern = f'{bias_type}.*?(?:example|instance|such as).*?([^.]+)'
            matches = re.findall(pattern, response, re.IGNORECASE)
            
            for match in matches:
                evidence_list.append(Evidence(
                    source=f"bias_audit_{bias_type.replace(' ', '_')}",
                    content=f"{bias_type}: {match.strip()}",
                    relevance_score=0.8
                ))
        
        # Also extract findings with → format
        arrow_pattern = r'(.+?)\s*→\s*(.+?)\s*→\s*(.+)'
        arrow_matches = re.findall(arrow_pattern, response)
        for finding, example, why in arrow_matches:
            if any(bias_word in finding.lower() for bias_word in ['bias', 'loaded', 'gendered']):
                evidence_list.append(Evidence(
                    source="bias_audit_structured",
                    content=f"{finding} → {example} → {why}",
                    relevance_score=0.85
                ))
        
        return evidence_list

    def _extract_position_length_analysis(self, response: str) -> List[str]:
        """Extract position and length bias analysis."""
        analyses = []
        
        # Extract position/length risk assessment
        import re
        risk_pattern = r'(?:position|length)\s*bias\s*risk:\s*(LOW|MEDIUM|HIGH)(?:\s*\+?\s*(.+?)(?:\n|$))?'
        matches = re.findall(risk_pattern, response, re.IGNORECASE)
        
        for risk_level, reason in matches:
            analysis_text = f"Bias risk: {risk_level}"
            if reason:
                analysis_text += f" - {reason.strip()}"
            analyses.append(analysis_text)
        
        # Look for explanations about judgment changes
        change_keywords = ['would change', 'might change', 'affect judgment', 'order matters']
        lines = response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in change_keywords):
                analyses.append(line.strip())
        
        return analyses[:4]

    def _extract_bias_issues(self, response: str) -> List[str]:
        """Extract specific bias issue tags."""
        issues = []
        
        # Standard bias issue format
        bias_tags = [
            'bias:position', 'bias:length', 'bias:confirmation',
            'bias:selection', 'bias:cultural'
        ]
        
        response_lower = response.lower()
        for tag in bias_tags:
            if tag.replace(':', ' ') in response_lower or tag in response_lower:
                issues.append(tag)
        
        # Also look for issues: line
        import re
        issues_pattern = r'issues:\s*\[([^\]]+)\]'
        match = re.search(issues_pattern, response, re.IGNORECASE)
        if match:
            issue_list = match.group(1)
            for issue in issue_list.split(','):
                issue_clean = issue.strip()
                if issue_clean and issue_clean not in issues:
                    issues.append(issue_clean)
        
        return issues[:6]
