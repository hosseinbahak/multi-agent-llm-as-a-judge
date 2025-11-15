# evaluate_realmistakes.py
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from sklearn.metrics import cohen_kappa_score, brier_score_loss
from scipy.stats import spearmanr, kendalltau
from statsmodels.stats.inter_rater import fleiss_kappa
import krippendorff
from loguru import logger
import sys
import os
from dotenv import load_dotenv
import time
import glob

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from the specific .env file
env_path = Path("/home/zeus/Projects/hb/multi_agent_llm_judge/.env")
load_dotenv(env_path)

from multi_agent_llm_judge.core.round_manager import RoundManager
from multi_agent_llm_judge.core.model_manager import ModelManager
from multi_agent_llm_judge.core.cache_manager import CacheManager, MemoryCache, DiskCache
from multi_agent_llm_judge.core.data_models import EvaluationRequest, Verdict, EvaluationResult, FinalJudgment
from multi_agent_llm_judge.providers.openrouter import OpenRouterClient
from multi_agent_llm_judge.utils.config_loader import load_config

class CalibrationDataCollector:
    """Collects evaluation data for future calibration training."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Different directories for data types
        self.evaluations_dir = self.data_dir / "evaluations"
        self.evaluations_dir.mkdir(exist_ok=True)
    
    def save_evaluation_with_ground_truth(self, result: EvaluationResult, ground_truth: bool, example_id: str) -> str:
        """
        Save evaluation result with ground truth label.
        
        Args:
            result: Evaluation result from jury system
            ground_truth: Whether the answer is actually correct
            example_id: ID of the example
            
        Returns:
            Unique ID of saved evaluation
        """
        if not result.success or not result.judgment:
            logger.error("Cannot save failed evaluation")
            return None
            
        # Generate unique ID for this evaluation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_id = f"eval_{timestamp}_{example_id}"
        
        # Extract evaluation data
        judgment = result.judgment
        jury_says_correct = judgment.is_correct == Verdict.CORRECT
        
        eval_data = {
            'eval_id': eval_id,
            'example_id': example_id,
            'timestamp': datetime.now().isoformat(),
            
            # Question and answer
            'question': judgment.question,
            'answer': judgment.answer,
            'context': judgment.context,
            
            # Ground truth
            'ground_truth': ground_truth,
            'is_answer_correct': ground_truth,  # For clarity
            
            # Jury results
            'jury_verdict': judgment.is_correct.value,
            'jury_says_correct': jury_says_correct,
            'verdict_correct': jury_says_correct == ground_truth,  # Whether jury was right
            'jury_confidence': judgment.raw_confidence,
            'jury_calibrated_confidence': judgment.calibrated_confidence,
            
            # Jury decision details
            'jury_decision': {
                'majority_verdict': judgment.jury_decision.majority_verdict.value,
                'consensus_level': judgment.jury_decision.consensus_level,
                'vote_distribution': judgment.jury_decision.vote_distribution,
                'weighted_confidence': judgment.jury_decision.weighted_confidence,
            },
            
            # Performance metrics
            'metrics': {
                'total_cost': judgment.total_cost,
                'total_tokens': judgment.total_tokens,
                'processing_time_ms': judgment.processing_time_ms,
                'models_used': judgment.models_used,
                'num_rounds': len(judgment.round_summaries),
                'num_agents': sum(len(rs.agent_analyses) for rs in judgment.round_summaries)
            },
            
            # Agent verdicts for inter-rater agreement
            'agent_verdicts': [
                {
                    'agent': a.agent_name,
                    'verdict': 1 if a.verdict == Verdict.CORRECT else 0
                }
                for rs in judgment.round_summaries
                for a in rs.agent_analyses
                if a.verdict is not None
            ]
        }
        
        # Save evaluation
        eval_path = self.evaluations_dir / f"{eval_id}.json"
        with open(eval_path, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation to {eval_path}")
        return eval_id

class RealMistakesEvaluator:
    def __init__(self, data_path: str, config_path: str = None):
        self.data_path = Path(data_path)
        self.results = []
        
        # Load configuration
        self.config = load_config(config_path) if config_path else load_config()
        
        # Initialize data collector
        self.collector = CalibrationDataCollector(Path("calibration_data/realmistakes"))
        
    async def initialize_pipeline(self):
        """Initialize the multi-agent judge pipeline."""
        # Initialize provider
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment variables")
        
        logger.info(f"Using OpenRouter API key: {api_key[:10]}...")
        
        self.provider = OpenRouterClient(api_key=api_key)
        
        # Initialize managers
        self.model_manager = ModelManager(provider=self.provider)
        await self.model_manager.initialize()
        
        # Initialize cache - using disk cache to persist between runs
        cache_backend = DiskCache(cache_dir=".cache/realmistakes")
        self.cache_manager = CacheManager(
            primary_backend=cache_backend,
            namespace="realmistakes_judge",
            stats_enabled=True
        )
        
        # Initialize round manager
        self.round_manager = RoundManager(
            config=self.config,
            model_manager=self.model_manager,
            cache_manager=self.cache_manager
        )
        
        logger.info("Pipeline initialized successfully")
    
    def load_jsonl_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL files matching the pattern."""
        pattern = "/home/zeus/Projects/hb/multi_agent_llm_judge/calibration_data/dataset/realmistakes/realmistake_json/*gpt4_with_gold.jsonl"
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        
        logger.info(f"Found {len(files)} JSONL files:")
        for file in files:
            logger.info(f"  - {file}")
        
        data = []
        for file_path in files:
            logger.info(f"Loading {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            example = json.loads(line)
                            # Add file info to metadata
                            if 'metadata' not in example:
                                example['metadata'] = {}
                            example['metadata']['source_file'] = Path(file_path).name
                            example['metadata']['line_number'] = line_num
                            
                            # Create a unique ID if not present
                            if 'id' not in example:
                                if 'metadata' in example and 'id' in example['metadata']:
                                    example['id'] = example['metadata']['id']
                                else:
                                    example['id'] = f"{Path(file_path).stem}_line_{line_num}"
                            
                            data.append(example)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing JSON in {file_path} at line {line_num}: {e}")
                            continue
        
        logger.info(f"Loaded total {len(data)} examples from {len(files)} files")
        return data
    
    def extract_detailed_round_analysis(self, judgment: FinalJudgment) -> Dict[str, Any]:
        """Extract detailed round-by-round analysis including agent verdicts and confidences."""
        round_details = []
        all_agent_details = []
        
        for round_idx, round_summary in enumerate(judgment.round_summaries):
            round_data = {
                'round_number': round_idx + 1,
                'agents': [],
                'round_summary': {
                    'total_agents': len(round_summary.agent_analyses),
                    'verdicts': {'correct': 0, 'incorrect': 0, 'uncertain': 0},
                    'avg_confidence': 0.0,
                    'models_used': []
                }
            }
            
            confidences = []
            
            for agent_analysis in round_summary.agent_analyses:
                # Safely get model name
                model_name = getattr(agent_analysis, 'model_name', 
                                    getattr(agent_analysis, 'model', 
                                            getattr(agent_analysis, 'agent_name', 'unknown')))
                
                # Safely get reasoning
                reasoning = getattr(agent_analysis, 'reasoning', 
                                   getattr(agent_analysis, 'rationale',
                                           getattr(agent_analysis, 'explanation',
                                                   getattr(agent_analysis, 'analysis',
                                                           getattr(agent_analysis, 'summary', 'No reasoning available')))))
                
                if reasoning and isinstance(reasoning, str):
                    reasoning_summary = reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
                else:
                    reasoning_summary = "No reasoning available"
                
                agent_detail = {
                    'agent_name': agent_analysis.agent_name,
                    'model': model_name,
                    'verdict': agent_analysis.verdict.value if agent_analysis.verdict else None,
                    'verdict_bool': agent_analysis.verdict == Verdict.CORRECT if agent_analysis.verdict else None,
                    'confidence': agent_analysis.confidence if hasattr(agent_analysis, 'confidence') and agent_analysis.confidence is not None else 0.0,
                    'reasoning_summary': reasoning_summary,
                    'cost': getattr(agent_analysis, 'cost', 0.0),
                    'tokens': getattr(agent_analysis, 'tokens', 0),
                    'processing_time_ms': getattr(agent_analysis, 'processing_time_ms', 0)
                }
                
                # Add to round data
                round_data['agents'].append(agent_detail)
                
                # Add to overall agent details with round info
                agent_detail_with_round = agent_detail.copy()
                agent_detail_with_round['round'] = round_idx + 1
                all_agent_details.append(agent_detail_with_round)
                
                # Update round summary
                if agent_analysis.verdict:
                    if agent_analysis.verdict == Verdict.CORRECT:
                        round_data['round_summary']['verdicts']['correct'] += 1
                    elif agent_analysis.verdict == Verdict.INCORRECT:
                        round_data['round_summary']['verdicts']['incorrect'] += 1
                    else:
                        round_data['round_summary']['verdicts']['uncertain'] += 1
                
                if hasattr(agent_analysis, 'confidence') and agent_analysis.confidence is not None:
                    confidences.append(agent_analysis.confidence)
                
                if model_name not in round_data['round_summary']['models_used']:
                    round_data['round_summary']['models_used'].append(model_name)
            
            # Calculate average confidence for this round
            if confidences:
                round_data['round_summary']['avg_confidence'] = sum(confidences) / len(confidences)
            
            round_details.append(round_data)
        
        return {
            'round_details': round_details,
            'all_agent_details': all_agent_details
        }
    
    async def evaluate_single_example(self, example: Dict[str, Any], retry_delay: float = 2.0) -> Dict[str, Any]:
        """Evaluate a single example with detailed output."""
        # Extract data from new format
        question = example["input"]  # Changed from "question"
        model_answer = example["llm_response"]  # Changed from "model_answer"
        ground_truth_verdict = example["gold_verdict"]  # Keep same - True/False for correctness
        difficulty = example["metadata"].get("difficulty", "unknown")  # Changed from "level"
        task_type = example.get("type", "unknown")  # Keep same
        example_id = example.get("id", "unknown")
        
        # For context, we can use the input directly or create minimal context
        context = f"Task: {task_type}\nDifficulty: {difficulty}"
        
        # Skip if no model answer
        if not model_answer or not model_answer.strip():
            logger.warning(f"Skipping example {example_id} - no model answer provided")
            return None
        
        # Create evaluation request
        request = EvaluationRequest(
            question=question,
            answer=model_answer,
            context=context,
            metadata={
                'id': example_id,
                'type': task_type,
                'difficulty': difficulty,
                'ground_truth_verdict': ground_truth_verdict,
                'error_label': example.get('error_label', 'unknown'),
                'error_categories': example.get('error_categories', []),
                'dataset': example['metadata'].get('dataset', 'realmistakes'),
                'task_source': example['metadata'].get('task_source', 'unknown'),
                'llm_response_model': example['metadata'].get('llm_response_model', 'unknown')
            }
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Run evaluation
                result = await self.round_manager.evaluate(request)
                
                if result.success and result.judgment:
                    # Save evaluation with ground truth
                    eval_id = self.collector.save_evaluation_with_ground_truth(
                        result=result,
                        ground_truth=ground_truth_verdict,
                        example_id=example_id
                    )
                    
                    judgment = result.judgment
                    
                    # Get pipeline verdict
                    pipeline_says_correct = judgment.is_correct == Verdict.CORRECT
                    
                    # Extract detailed round and agent analysis
                    detailed_analysis = self.extract_detailed_round_analysis(judgment)
                    
                    # Collect jury verdicts for analysis
                    jury_verdicts = []
                    for agent_detail in detailed_analysis['all_agent_details']:
                        if agent_detail['verdict_bool'] is not None:
                            jury_verdicts.append({
                                'round': agent_detail['round'],
                                'agent': agent_detail['agent_name'],
                                'model': agent_detail['model'],
                                'verdict': agent_detail['verdict'],
                                'verdict_bool': agent_detail['verdict_bool'],
                                'confidence': agent_detail['confidence']
                            })
                    
                    result_data = {
                        # Basic info
                        'id': example_id,
                        'question': question,
                        'model_answer': model_answer,
                        'task_type': task_type,
                        'difficulty': difficulty,
                        'error_label': example.get('error_label', 'unknown'),
                        'error_categories': example.get('error_categories', []),
                        'dataset': example['metadata'].get('dataset', 'realmistakes'),
                        'task_source': example['metadata'].get('task_source', 'unknown'),
                        'llm_response_model': example['metadata'].get('llm_response_model', 'unknown'),
                        
                        # Ground truth and pipeline results
                        'ground_truth_verdict': ground_truth_verdict,
                        'pipeline_verdict': pipeline_says_correct,
                        'verdict_correct': ground_truth_verdict == pipeline_says_correct,
                        
                        # Confidence and consensus
                        'total_confidence': judgment.raw_confidence,
                        'calibrated_confidence': judgment.calibrated_confidence if judgment.calibrated_confidence else judgment.raw_confidence,
                        'consensus_level': judgment.jury_decision.consensus_level,
                        'vote_distribution': judgment.jury_decision.vote_distribution,
                        'weighted_confidence': judgment.jury_decision.weighted_confidence,
                        
                        # Round-by-round analysis
                        'round_details': detailed_analysis['round_details'],
                        'agent_details': detailed_analysis['all_agent_details'],
                        'jury_verdicts': jury_verdicts,
                        
                        # Performance metrics
                        'processing_time_ms': judgment.processing_time_ms,
                        'total_cost': judgment.total_cost,
                        'total_tokens': judgment.total_tokens,
                        'models_used': judgment.models_used,
                        'num_rounds': len(judgment.round_summaries),
                        'num_agents': len(detailed_analysis['all_agent_details']),
                        
                        # Evaluation ID for tracking
                        'eval_id': eval_id
                    }
                    
                    return result_data
                else:
                    logger.error(f"Evaluation failed for example {example_id}: {result.error}")
                    return None
                    
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    # Rate limit error, wait and retry
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Error evaluating example {example_id}: {e}")
                    if attempt == max_retries - 1:
                        import traceback
                        traceback.print_exc()
                    return None
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate all required metrics."""
        # Extract data
        verdict_corrects = [r['verdict_correct'] for r in results]
        ground_truth_verdicts = [r['ground_truth_verdict'] for r in results]
        pipeline_verdicts = [r['pipeline_verdict'] for r in results]
        confidences = [r['total_confidence'] for r in results]
        
        # Convert to numpy arrays
        verdict_corrects = np.array(verdict_corrects)
        ground_truth_verdicts = np.array(ground_truth_verdicts, dtype=int)
        pipeline_verdicts = np.array(pipeline_verdicts, dtype=int)
        confidences = np.array(confidences)
        
        # Basic metrics
        n_total = len(results)
        verdict_accuracy = float(np.mean(verdict_corrects))
        
        # Cohen's kappa
        cohen_kappa = cohen_kappa_score(ground_truth_verdicts, pipeline_verdicts)
        
        # Spearman correlation
        spearman_r, spearman_p = spearmanr(confidences, verdict_corrects)
        
        # Kendall's tau
        kendall_tau, kendall_p = kendalltau(confidences, verdict_corrects)
        
        # Brier score
        brier = brier_score_loss(ground_truth_verdicts, confidences)
        
        # ECE
        ece = self.calculate_ece(confidences, verdict_corrects)
        
        # Calculate inter-rater agreement metrics
        inter_rater_metrics = self.calculate_inter_rater_metrics(results)
        
        # Breakdown by task type and difficulty (changed from question type and level)
        type_breakdown = {}
        difficulty_breakdown = {}
        
        for result in results:
            t_type = result['task_type']
            difficulty = result['difficulty']
            
            if t_type not in type_breakdown:
                type_breakdown[t_type] = {'count': 0, 'accuracy': []}
            type_breakdown[t_type]['count'] += 1
            type_breakdown[t_type]['accuracy'].append(result['verdict_correct'])
            
            if difficulty not in difficulty_breakdown:
                difficulty_breakdown[difficulty] = {'count': 0, 'accuracy': []}
            difficulty_breakdown[difficulty]['count'] += 1
            difficulty_breakdown[difficulty]['accuracy'].append(result['verdict_correct'])
        
        # Calculate averages
        for t_type in type_breakdown:
            type_breakdown[t_type]['accuracy'] = float(np.mean(type_breakdown[t_type]['accuracy']))
        
        for difficulty in difficulty_breakdown:
            difficulty_breakdown[difficulty]['accuracy'] = float(np.mean(difficulty_breakdown[difficulty]['accuracy']))
        
        metrics = {
            "N_total": n_total,
            "Verdict_Accuracy": verdict_accuracy,
            "Cohen_kappa": float(cohen_kappa),
            "Spearman_r": float(spearman_r) if not np.isnan(spearman_r) else 0.0,
            "Spearman_p": float(spearman_p) if not np.isnan(spearman_p) else 1.0,
            "Kendall_tau_b": float(kendall_tau) if not np.isnan(kendall_tau) else 0.0,
            "Kendall_p": float(kendall_p) if not np.isnan(kendall_p) else 1.0,
            "Brier": float(brier),
            "ECE@10": float(ece),
            "Inter_rater_agreement": inter_rater_metrics,
            "Type_breakdown": type_breakdown,
            "Difficulty_breakdown": difficulty_breakdown  # Changed from Level_breakdown
        }
        
        return metrics
    
    def calculate_inter_rater_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate inter-rater agreement metrics."""
        try:
            # Collect all agent verdicts for each example
            all_verdicts = []
            
            for result in results:
                example_verdicts = []
                for agent_detail in result['agent_details']:
                    if agent_detail['verdict_bool'] is not None:
                        example_verdicts.append(1 if agent_detail['verdict_bool'] else 0)
                
                if len(example_verdicts) >= 2:  # Need at least 2 raters
                    all_verdicts.append(example_verdicts)
            
            if len(all_verdicts) < 2:
                return {"error": "Insufficient data for inter-rater agreement"}
            
            # Calculate Fleiss' kappa if we have consistent number of raters
            rater_counts = [len(v) for v in all_verdicts]
            if len(set(rater_counts)) == 1:  # All examples have same number of raters
                # Prepare data for Fleiss' kappa
                n_raters = rater_counts[0]
                fleiss_data = []
                
                for verdicts in all_verdicts:
                    # Count votes for each category (0=incorrect, 1=correct)
                    count_incorrect = verdicts.count(0)
                    count_correct = verdicts.count(1)
                    fleiss_data.append([count_incorrect, count_correct])
                
                fleiss_k = fleiss_kappa(np.array(fleiss_data))
                
                return {
                    "fleiss_kappa": float(fleiss_k),
                    "n_examples": len(all_verdicts),
                    "n_raters": n_raters
                }
            else:
                return {
                    "error": "Inconsistent number of raters across examples",
                    "rater_counts": rater_counts
                }
                
        except Exception as e:
            logger.error(f"Error calculating inter-rater metrics: {e}")
            return {"error": str(e)}
    
    def calculate_ece(self, confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_confidence = np.mean(confidences[in_bin])
                bin_weight = np.sum(in_bin) / len(confidences)
                ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
        
        return ece
    
    async def run_evaluation(self, max_examples: int = None, delay_between_examples: float = 2.0):
        """Run evaluation on RealMistakes dataset, one question at a time."""
        # Load data from JSONL files
        data = self.load_jsonl_data()
        
        # Limit examples if specified
        if max_examples:
            data = data[:max_examples]
        
        logger.info(f"Will evaluate {len(data)} examples")
        
        # Initialize pipeline
        await self.initialize_pipeline()
        
        # Evaluate one by one
        results = []
        for idx, example in enumerate(data):
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating example {idx + 1}/{len(data)} - ID: {example['id']}")
            logger.info(f"Question: {example['input'][:100]}...")
            logger.info(f"Type: {example.get('type', 'unknown')}, Difficulty: {example['metadata'].get('difficulty', 'unknown')}")
            logger.info(f"Gold verdict: {example['gold_verdict']}")
            logger.info(f"Error label: {example.get('error_label', 'unknown')}")
            
            # Evaluate single example
            result = await self.evaluate_single_example(example)
            
            if result:
                results.append(result)
                logger.info(f"✓ Evaluation complete")
                logger.info(f"  Pipeline verdict: {result['pipeline_verdict']} (Ground truth: {result['ground_truth_verdict']})")
                logger.info(f"  Verdict correct: {result['verdict_correct']}")
                logger.info(f"  Confidence: {result['total_confidence']:.2%}")
                logger.info(f"  Consensus: {result['consensus_level']:.2%}")
                logger.info(f"  Agents: {result['num_agents']} across {result['num_rounds']} rounds")
                
                # Show round-by-round details
                logger.info("  Round-by-round analysis:")
                for round_detail in result['round_details']:
                    round_num = round_detail['round_number']
                    round_summary = round_detail['round_summary']
                    logger.info(f"    Round {round_num}: {round_summary['total_agents']} agents, avg conf: {round_summary['avg_confidence']:.2%}")
                    logger.info(f"      Verdicts: {round_summary['verdicts']}")
                    
                    # Show individual agent verdicts for this round
                    for agent in round_detail['agents']:
                        conf_str = f"{agent['confidence']:.2%}" if agent['confidence'] else "N/A"
                        logger.info(f"        {agent['agent_name']} ({agent['model']}): {agent['verdict']} (conf: {conf_str})")
                        
            else:
                logger.warning(f"✗ Failed to evaluate example {example['id']}")
            
            # Add delay between examples to avoid rate limits
            if idx < len(data) - 1:  # Don't delay after the last example
                logger.info(f"Waiting {delay_between_examples}s before next example...")
                await asyncio.sleep(delay_between_examples)
        
        # Calculate metrics
        if results:
            metrics = self.calculate_metrics(results)
            
            # Save results
            output_dir = Path("evaluation_results")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed results
            results_path = output_dir / f"realmistakes_detailed_results_{timestamp}.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'timestamp': timestamp,
                        'n_examples': len(data),
                        'n_evaluated': len(results)
                    },
                    'metrics': metrics,
                    'results': results
                }, f, indent=2)
            
            # Save metrics summary
            metrics_path = output_dir / f"realmistakes_metrics_{timestamp}.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)
            
            # Print metrics
            print("\n" + "="*80)
            print("EVALUATION METRICS")
            print("="*80)
            for metric_name, value in metrics.items():
                if metric_name in ['Type_breakdown', 'Difficulty_breakdown', 'Inter_rater_agreement']:
                    print(f"{metric_name}:")
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            print(f"  {sub_key}: {sub_value}")
                elif isinstance(value, float):
                    print(f"{metric_name}: {value:.4f}")
                else:
                    print(f"{metric_name}: {value}")
            print("="*80)
            
            logger.info(f"Results saved to {results_path}")
            logger.info(f"Metrics saved to {metrics_path}")
            logger.info(f"Calibration data saved to {self.collector.data_dir}")
        else:
            logger.error("No valid results to calculate metrics")
        
        # Cleanup
        await self.round_manager.agent_manager.cleanup()
        await self.provider.close()

async def main():
    """Main entry point."""
    # Create evaluator - no need to specify data_path since we're using glob pattern
    evaluator = RealMistakesEvaluator(data_path="/home/zeus/Projects/hb/multi_agent_llm_judge/calibration_data/dataset/realmistakes/realmistake_json/math_word_problem_generation__gpt4_with_gold.jsonl")
    
    # Run evaluation one by one with 2-second delay between examples
    # Start with a few examples for testing, then increase
    await evaluator.run_evaluation(max_examples=1000, delay_between_examples=2.0)

if __name__ == "__main__":
    asyncio.run(main())
