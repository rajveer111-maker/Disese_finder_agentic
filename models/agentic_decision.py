import numpy as np
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgenticDecisionSystem:
    """
    Agentic system that intelligently decides which model to use based on input data characteristics.
    """
    
    def __init__(self):
        self.model_capabilities = {
            'bci2a_crdae': {
                'data_types': ['eeg', 'csv', 'txt'],
                'channels_range': (16, 64),
                'sample_rate_range': (100, 1000),
                'duration_range': (1, 10),  # seconds
                'use_cases': ['motor_imagery', 'bci', 'rehabilitation'],
                'confidence_threshold': 0.6
            },
            'eeg_pd': {
                'data_types': ['eeg', 'csv', 'txt', 'edf'],
                'channels_range': (16, 64),
                'sample_rate_range': (100, 1000),
                'duration_range': (2, 30),  # seconds
                'use_cases': ['parkinsons', 'neurological', 'clinical'],
                'confidence_threshold': 0.7
            },
            'neuroformer': {
                'data_types': ['eeg', 'csv', 'txt', 'npy', 'edf'],
                'channels_range': (10, 64),
                'sample_rate_range': (100, 1000),
                'duration_range': (2, 30),  # seconds
                'use_cases': ['alzheimer', 'dementia', 'ftd', 'cognitive', 'neurological'],
                'confidence_threshold': 0.75
            },
            'nhrn_pd': {
                'data_types': ['eeg', 'csv', 'txt', 'npy', 'edf'],
                'channels_range': (8, 64),
                'sample_rate_range': (100, 1000),
                'duration_range': (2, 30),  # seconds
                'use_cases': ['parkinsons', 'neurological', 'clinical', 'resonance'],
                'confidence_threshold': 0.8
            }
        }
        
        self.decision_rules = [
            self._rule_file_type,
            self._rule_data_characteristics,
            self._rule_use_case_context,
            self._rule_confidence_estimation
        ]
    
    def decide_model(self, data: np.ndarray, file_type: str, 
                    context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Decide which model to use based on input data and context.
        
        Args:
            data: Input data array
            file_type: Type of input file
            context: Additional context information
            
        Returns:
            Dictionary containing decision results
        """
        if context is None:
            context = {}
        
        # Calculate data characteristics
        data_characteristics = self._analyze_data_characteristics(data)
        
        # Apply decision rules
        scores = {key: 0.0 for key in self.model_capabilities.keys()}
        reasoning = []
        
        for rule in self.decision_rules:
            rule_scores, rule_reasoning = rule(data, file_type, data_characteristics, context)
            for k, v in rule_scores.items():
                scores[k] += v
            reasoning.extend(rule_reasoning)
            
        num_rules = len(self.decision_rules)
        for k in scores:
            scores[k] /= num_rules
        
        # Select best model
        best_model = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_model]
        
        # Generate final reasoning
        final_reasoning = self._generate_final_reasoning(
            best_model, confidence, reasoning, data_characteristics
        )
        
        return {
            'selected_model': best_model,
            'confidence': confidence,
            'reasoning': final_reasoning,
            'all_scores': scores,
            'data_characteristics': data_characteristics
        }
    
    def _analyze_data_characteristics(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of the input data."""
        characteristics = {
            'shape': data.shape,
            'channels': data.shape[1] if len(data.shape) > 1 else 1,
            'samples': data.shape[0],
            'duration_estimate': data.shape[0] / 250,  # Assuming 250Hz sampling rate
            'data_type': 'eeg',  # Default assumption
            'has_variation': np.std(data) > 0.1,
            'signal_quality': self._assess_signal_quality(data)
        }
        
        return characteristics
    
    def _assess_signal_quality(self, data: np.ndarray) -> str:
        """Assess the quality of the EEG signal."""
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        # Calculate signal quality metrics
        signal_power = np.mean(np.var(data, axis=0))
        noise_level = np.mean(np.abs(np.diff(data, axis=0)))
        
        if signal_power > 1.0 and noise_level < 0.5:
            return 'high'
        elif signal_power > 0.5 and noise_level < 1.0:
            return 'medium'
        else:
            return 'low'
    
    def _rule_file_type(self, data: np.ndarray, file_type: str, 
                       characteristics: Dict[str, Any], context: Dict[str, Any]) -> tuple:
        """Rule based on file type compatibility."""
        scores = {}
        reasoning = []
        
        for model_key, capabilities in self.model_capabilities.items():
            if file_type.lower() in capabilities['data_types']:
                scores[model_key] = 0.8
                reasoning.append(f"OK {model_key} supports {file_type} files")
            else:
                scores[model_key] = 0.2
                reasoning.append(f"WARNING {model_key} has limited support for {file_type} files")
        
        return scores, reasoning
    
    def _rule_data_characteristics(self, data: np.ndarray, file_type: str,
                                 characteristics: Dict[str, Any], context: Dict[str, Any]) -> tuple:
        """Rule based on data characteristics compatibility."""
        scores = {}
        reasoning = []
        
        channels = characteristics['channels']
        duration = characteristics['duration_estimate']
        signal_quality = characteristics['signal_quality']
        
        for model_key, capabilities in self.model_capabilities.items():
            score = 0.5  # Base score
            
            # Check channel compatibility
            min_channels, max_channels = capabilities['channels_range']
            if min_channels <= channels <= max_channels:
                score += 0.2
                reasoning.append(f"OK {model_key} supports {channels} channels")
            else:
                score -= 0.1
                reasoning.append(f"WARNING {model_key} expects {min_channels}-{max_channels} channels, got {channels}")
            
            # Check duration compatibility
            min_duration, max_duration = capabilities['duration_range']
            if min_duration <= duration <= max_duration:
                score += 0.2
                reasoning.append(f"OK {model_key} works well with {duration:.1f}s duration")
            else:
                score -= 0.1
                reasoning.append(f"WARNING {model_key} expects {min_duration}-{max_duration}s, got {duration:.1f}s")
            
            # Check signal quality
            if signal_quality == 'high':
                score += 0.1
                reasoning.append(f"OK High quality signal detected")
            elif signal_quality == 'low':
                score -= 0.1
                reasoning.append(f"WARNING Low quality signal detected")
            
            scores[model_key] = max(0.0, min(1.0, score))
        
        return scores, reasoning
    
    def _rule_use_case_context(self, data: np.ndarray, file_type: str,
                             characteristics: Dict[str, Any], context: Dict[str, Any]) -> tuple:
        """Rule based on use case context."""
        scores = {}
        reasoning = []
        
        # Check for keywords in context that might indicate use case
        context_text = str(context).lower()
        
        for model_key, capabilities in self.model_capabilities.items():
            score = 0.5  # Base score
            
            for use_case in capabilities['use_cases']:
                if use_case in context_text:
                    score += 0.3
                    reasoning.append(f"OK Context suggests {use_case} use case for {model_key}")
                    break
            
            scores[model_key] = score
        
        return scores, reasoning
    
    def _rule_confidence_estimation(self, data: np.ndarray, file_type: str,
                                  characteristics: Dict[str, Any], context: Dict[str, Any]) -> tuple:
        """Rule based on estimated confidence for each model."""
        scores = {}
        reasoning = []
        
        for model_key, capabilities in self.model_capabilities.items():
            # Estimate confidence based on data quality and compatibility
            base_confidence = capabilities['confidence_threshold']
            
            # Adjust based on signal quality
            signal_quality = characteristics['signal_quality']
            if signal_quality == 'high':
                confidence = base_confidence + 0.2
            elif signal_quality == 'medium':
                confidence = base_confidence
            else:
                confidence = base_confidence - 0.2
            
            # Adjust based on data characteristics match
            channels = characteristics['channels']
            min_channels, max_channels = capabilities['channels_range']
            if min_channels <= channels <= max_channels:
                confidence += 0.1
            
            scores[model_key] = max(0.0, min(1.0, confidence))
            reasoning.append(f"Estimated confidence for {model_key}: {confidence:.2f}")
        
        return scores, reasoning
    
    def _generate_final_reasoning(self, selected_model: str, confidence: float,
                                reasoning: List[str], characteristics: Dict[str, Any]) -> str:
        """Generate final reasoning text."""
        reasoning_parts = [
            f"Selected {selected_model} model with {confidence:.1%} confidence.",
            f"Data characteristics: {characteristics['channels']} channels, "
            f"{characteristics['duration_estimate']:.1f}s duration, "
            f"{characteristics['signal_quality']} quality signal.",
            "Key factors:"
        ]
        
        # Add relevant reasoning points
        relevant_reasons = [r for r in reasoning if selected_model in r or '✅' in r]
        reasoning_parts.extend(relevant_reasons[:3])  # Limit to top 3 reasons
        
        return " ".join(reasoning_parts)
    
    def get_model_recommendations(self, use_case: str) -> List[Dict[str, Any]]:
        """Get model recommendations for a specific use case."""
        recommendations = []
        
        for model_key, capabilities in self.model_capabilities.items():
            if use_case.lower() in [uc.lower() for uc in capabilities['use_cases']]:
                recommendations.append({
                    'model': model_key,
                    'suitability': 'high',
                    'description': capabilities.get('description', ''),
                    'confidence_threshold': capabilities['confidence_threshold']
                })
        
        return recommendations
