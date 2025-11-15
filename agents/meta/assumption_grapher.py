# -*- coding: utf-8 -*-
"""
AssumptionGrapherAgent — PARC-style mapping of assumptions to reasoning steps with acceptability ratings.
"""

from typing import List, Optional
import re
from ..base_agent import BaseAgent
from ...core.data_models import AgentAnalysis, AgentType, Evidence, Verdict

class AssumptionGrapherAgent(BaseAgent):
    """Agent that maps assumptions to reasoning steps."""

    agent_type = AgentType.ASSUMPTION_GRAPHER
    default_system_prompt = """You map assumptions like a PARC-style DAG.

Tasks:
1) List assumptions in three buckets:
   - Explicit (stated)
   - Implicit (unstated but required)
   - External (needs outside knowledge/lookup)
2) Link each assumption to the step(s) it supports.
3) Rate each assumption: ACCEPTABLE / NEEDS_EVIDENCE / INVALID.
4) Identify the minimal assumption set that makes the reasoning go through.

Output:
- Assumption map: [assumption → type → linked steps → rating]
- Minimal assumption set
- Issues: [parametric_knowledge_error, logical_error]
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

Answer to analyze: {answer}

"""
        if context:
            prompt += f"Context: {context}\n\n"

        prompt += """Map all assumptions following the PARC-style format specified."""

        return prompt

    def _parse_response(
        self,
        response_text: str,
        model_id: str,
        tokens_used: int,
        cost: float,
        processing_time_ms: int
    ) -> AgentAnalysis:
        """Parse the assumption mapping response."""
        # Extract verdict based on assumption validity
        verdict = self._determine_assumption_verdict(response_text)
        
        # Extract confidence
        confidence = self._extract_confidence(response_text)
        
        # Extract assumption mappings
        evidence = self._extract_assumption_mappings(response_text)
        
        # Extract minimal assumption set
        reasoning_steps = self._extract_minimal_set(response_text)
        
        # Extract assumptions as separate field
        assumptions = self._extract_categorized_assumptions(response_text)
        
        # Extract issues
        limitations = self._extract_assumption_issues(response_text)

        return AgentAnalysis(
            agent_type=self.agent_type,
            agent_name=self.name,
            model_used=model_id,
            analysis=response_text,
            verdict=verdict,
            confidence=confidence,
            evidence=evidence,
            reasoning_steps=reasoning_steps,
            assumptions=assumptions,
            limitations=limitations,
            tokens_used=tokens_used,
            cost=cost,
            processing_time_ms=processing_time_ms
        )

    def _determine_assumption_verdict(self, response: str) -> Verdict:
        """Determine verdict based on assumption analysis."""
        response_lower = response.lower()
        
        # Count assumption ratings
        invalid_count = response_lower.count('invalid')
        needs_evidence_count = response_lower.count('needs_evidence') + response_lower.count('needs evidence')
        acceptable_count = response_lower.count('acceptable')
        
        # Verdict logic based on assumptions
        if invalid_count > 0:
            return Verdict.INCORRECT
        elif needs_evidence_count > 2:
            return Verdict.UNCERTAIN
        elif acceptable_count > needs_evidence_count:
            # Also check explicit verdict
            verdict_bool = self._extract_verdict(response)
            if verdict_bool is True:
                return Verdict.CORRECT
            elif verdict_bool is False:
                return Verdict.INCORRECT
            else:
                return Verdict.UNCERTAIN
        else:
            return Verdict.UNCERTAIN

    def _extract_assumption_mappings(self, response: str) -> List[Evidence]:
        """Extract assumption to step mappings."""
        evidence_list = []
        
        # Pattern for assumption → type → steps → rating
        mapping_pattern = r'(.+?)\s*→\s*(explicit|implicit|external)\s*→\s*(.+?)\s*→\s*(ACCEPTABLE|NEEDS_EVIDENCE|INVALID|NEEDS EVIDENCE)'
        matches = re.findall(mapping_pattern, response, re.IGNORECASE)
        
        for assumption, type_, steps, rating in matches:
            rating_score = {'ACCEPTABLE': 0.9, 'NEEDS_EVIDENCE': 0.5, 'INVALID': 0.2}.get(rating.upper(), 0.5)
            
            evidence_list.append(Evidence(
                source=f"assumption_{type_.lower()}",
                content=f"{assumption.strip()} [{rating}] → {steps.strip()}",
                relevance_score=rating_score
            ))
        
        # Also extract simpler formats
        bullet_pattern = r'[-•*]\s*(.+?):\s*(.+)'
        bullet_matches = re.findall(bullet_pattern, response)
        for assumption, details in bullet_matches:
            if any(word in assumption.lower() for word in ['assumption', 'assume', 'premise']):
                evidence_list.append(Evidence(
                    source="assumption_identified",
                    content=f"{assumption}: {details}",
                    relevance_score=0.7
                ))
        
        return evidence_list

    def _extract_minimal_set(self, response: str) -> List[str]:
        """Extract the minimal assumption set."""
        minimal_set = []
        
        # Look for minimal set section
        minimal_section = re.search(
            r'minimal\s+assumption\s+set:?(.*?)(?:issues:|final verdict:|confidence|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        
        if minimal_section:
            section_text = minimal_section.group(1)
            # Extract items
            items = re.findall(r'[-•*]\s*(.+)|(\d+)[.)\s]+(.+)', section_text)
            for item in items:
                item_text = ' '.join(part for part in item if part).strip()
                if len(item_text) > 10:
                    minimal_set.append(item_text)
        
        return minimal_set[:5]

    def _extract_categorized_assumptions(self, response: str) -> List[str]:
        """Extract assumptions by category."""
        assumptions = []
        
        # Categories to look for
        categories = ['explicit', 'implicit', 'external']
        
        for category in categories:
            # Find category section
            pattern = f'{category}(?:\s+assumptions)?:?(.*?)(?:explicit|implicit|external|link|output|$)'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            
            if match:
                section_text = match.group(1)
                # Extract assumptions from section
                items = re.findall(r'[-•*]\s*(.+)|(\d+)[.)\s]+(.+)', section_text)
                for item in items:
                    item_text = ' '.join(part for part in item if part).strip()
                    if len(item_text) > 5:
                        assumptions.append(f"{category.capitalize()}: {item_text}")
        
        return assumptions[:7]

    def _extract_assumption_issues(self, response: str) -> List[str]:
        """Extract issues related to assumptions."""
        issues = []
        
        # Issue keywords
        issue_keywords = [
            'parametric_knowledge_error', 'logical_error',
            'unsupported assumption', 'circular reasoning',
            'missing premise', 'contradictory assumption'
        ]
        
        lines = response.split('\n')
        for line in lines:
            line_lower = line.lower()
            for keyword in issue_keywords:
                if keyword in line_lower:
                    issues.append(line.strip())
                    break
        
        return issues[:4]

    def _map_verdict(self, verdict_str: Optional[str], positive_term: str = "SOLID") -> Verdict:
        """Map custom verdict terms to standard Verdict enum."""
        if not verdict_str:
            return Verdict.UNCERTAIN
        
        verdict_upper = verdict_str.upper()
        if verdict_upper == positive_term.upper() or verdict_upper == "CORRECT":
            return Verdict.CORRECT
        elif verdict_upper == "SHAKY" or verdict_upper == "INCORRECT":
            return Verdict.INCORRECT
        else:
            return Verdict.UNCERTAIN
