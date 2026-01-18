"""
Domain-Specific Evaluators for Expert Profiling

Provides evaluation functions for various domains beyond Math/Agentic/Planning.
"""
from __future__ import annotations

import re
import json
from typing import List, Dict, Any


class CodeEvaluator:
    """Evaluate code generation and understanding."""

    def evaluate(self, predictions: List[str], references: List[str]) -> float:
        """
        Score code quality based on syntax validity, structure, and patterns.

        Args:
            predictions: Generated code snippets
            references: Reference code snippets

        Returns:
            Score between 0-1
        """
        if not predictions:
            return 0.0

        scores = []
        for pred in predictions:
            score = 0.0

            # Check for code-like patterns
            if re.search(r'(def |class |import |from |function |const |let |var )', pred):
                score += 0.2

            # Check for proper indentation
            if '\n    ' in pred or '\n\t' in pred:
                score += 0.2

            # Check for common code structures
            if re.search(r'[\{\}\[\]\(\)]', pred):
                score += 0.2

            # Check for comments
            if re.search(r'(#|//|/\*|\*/)', pred):
                score += 0.1

            # Check for variable assignments
            if re.search(r'[a-zA-Z_]\w*\s*=', pred):
                score += 0.15

            # Check for control flow
            if re.search(r'(if|else|for|while|return)', pred):
                score += 0.15

            scores.append(min(score, 1.0))

        return sum(scores) / len(scores)


class ScienceEvaluator:
    """Evaluate scientific reasoning and knowledge."""

    def evaluate(self, predictions: List[str], references: List[str]) -> float:
        """
        Score scientific content based on terminology, units, and reasoning.

        Args:
            predictions: Generated scientific text
            references: Reference scientific text

        Returns:
            Score between 0-1
        """
        if not predictions:
            return 0.0

        scores = []
        for pred in predictions:
            score = 0.0

            # Check for scientific terminology
            sci_terms = [
                'hypothesis', 'experiment', 'theory', 'observation', 'measurement',
                'molecule', 'atom', 'cell', 'organism', 'species', 'evolution',
                'energy', 'force', 'mass', 'velocity', 'temperature', 'pressure',
                'chemical', 'biological', 'physical', 'quantum', 'reaction',
            ]
            term_count = sum(1 for term in sci_terms if term.lower() in pred.lower())
            score += min(term_count * 0.1, 0.4)

            # Check for units
            if re.search(r'\d+\s*(m|kg|s|K|mol|A|cd|m/s|J|N|Pa|Hz|W)', pred):
                score += 0.2

            # Check for scientific notation
            if re.search(r'\d+\.?\d*\s*[×x]\s*10\^?-?\d+', pred):
                score += 0.1

            # Check for equations or formulas
            if re.search(r'[=+\-*/^]', pred) and any(c.isdigit() for c in pred):
                score += 0.2

            # Check for reasoning words
            reasoning_words = ['because', 'therefore', 'thus', 'hence', 'as a result', 'due to']
            if any(word in pred.lower() for word in reasoning_words):
                score += 0.1

            scores.append(min(score, 1.0))

        return sum(scores) / len(scores)


class ReasoningEvaluator:
    """Evaluate logical reasoning and inference."""

    def evaluate(self, predictions: List[str], references: List[str]) -> float:
        """
        Score reasoning quality based on logic, structure, and coherence.

        Args:
            predictions: Generated reasoning text
            references: Reference reasoning text

        Returns:
            Score between 0-1
        """
        if not predictions:
            return 0.0

        scores = []
        for pred in predictions:
            score = 0.0

            # Check for reasoning connectives
            connectives = [
                'therefore', 'thus', 'hence', 'consequently', 'because', 'since',
                'if', 'then', 'assuming', 'given that', 'implies', 'follows that',
                'must be', 'cannot be', 'either', 'neither',
            ]
            connective_count = sum(1 for conn in connectives if conn.lower() in pred.lower())
            score += min(connective_count * 0.1, 0.4)

            # Check for structured arguments (numbered lists)
            if re.search(r'(1\.|2\.|3\.|first|second|third)', pred.lower()):
                score += 0.2

            # Check for premise-conclusion structure
            if 'premise' in pred.lower() or 'conclusion' in pred.lower():
                score += 0.2

            # Check for evidence markers
            evidence_markers = ['evidence', 'proof', 'support', 'demonstrates', 'shows that']
            if any(marker in pred.lower() for marker in evidence_markers):
                score += 0.1

            # Check for logical consistency (no contradictions)
            if not ('but' in pred.lower() and 'however' in pred.lower()):
                score += 0.1

            scores.append(min(score, 1.0))

        return sum(scores) / len(scores)


class VisionEvaluator:
    """Evaluate vision and spatial reasoning (text-based proxy)."""

    def evaluate(self, predictions: List[str], references: List[str]) -> float:
        """
        Score vision-related understanding based on spatial/visual language.

        Args:
            predictions: Generated vision-related text
            references: Reference vision-related text

        Returns:
            Score between 0-1
        """
        if not predictions:
            return 0.0

        scores = []
        for pred in predictions:
            score = 0.0

            # Check for spatial terms
            spatial_terms = [
                'left', 'right', 'top', 'bottom', 'above', 'below', 'center',
                'corner', 'edge', 'side', 'front', 'back', 'near', 'far',
                'horizontal', 'vertical', 'diagonal', 'parallel', 'perpendicular',
            ]
            term_count = sum(1 for term in spatial_terms if term.lower() in pred.lower())
            score += min(term_count * 0.1, 0.3)

            # Check for visual descriptors
            visual_terms = [
                'color', 'shape', 'size', 'bright', 'dark', 'red', 'blue', 'green',
                'circle', 'square', 'triangle', 'line', 'curve', 'pattern',
                'image', 'picture', 'visual', 'appears', 'looks like',
            ]
            visual_count = sum(1 for term in visual_terms if term.lower() in pred.lower())
            score += min(visual_count * 0.1, 0.3)

            # Check for counting objects
            if re.search(r'(there are|contains|has)\s+\d+', pred.lower()):
                score += 0.2

            # Check for comparative descriptions
            if re.search(r'(larger|smaller|bigger|closer|farther|more|less) than', pred.lower()):
                score += 0.2

            scores.append(min(score, 1.0))

        return sum(scores) / len(scores)


class RoboticsEvaluator:
    """Evaluate robotics and control understanding."""

    def evaluate(self, predictions: List[str], references: List[str]) -> float:
        """
        Score robotics knowledge based on terminology and concepts.

        Args:
            predictions: Generated robotics text
            references: Reference robotics text

        Returns:
            Score between 0-1
        """
        if not predictions:
            return 0.0

        scores = []
        for pred in predictions:
            score = 0.0

            # Check for robotics terms
            robotics_terms = [
                'robot', 'actuator', 'sensor', 'motor', 'gripper', 'manipulator',
                'joint', 'link', 'kinematics', 'trajectory', 'path planning',
                'obstacle', 'collision', 'navigation', 'localization', 'mapping',
                'control', 'feedback', 'servo', 'encoder', 'lidar', 'camera',
            ]
            term_count = sum(1 for term in robotics_terms if term.lower() in pred.lower())
            score += min(term_count * 0.1, 0.4)

            # Check for movement commands
            movement_terms = ['move', 'rotate', 'turn', 'stop', 'go', 'forward', 'backward', 'grasp', 'release']
            if any(term in pred.lower() for term in movement_terms):
                score += 0.2

            # Check for coordinate/position references
            if re.search(r'(x|y|z)\s*[:=]\s*[-]?\d+', pred.lower()):
                score += 0.2

            # Check for control parameters
            if re.search(r'(speed|velocity|acceleration|angle|position|distance)', pred.lower()):
                score += 0.2

            scores.append(min(score, 1.0))

        return sum(scores) / len(scores)


class MultilingualEvaluator:
    """Evaluate multilingual capabilities."""

    def evaluate(self, predictions: List[str], references: List[str]) -> float:
        """
        Score multilingual content based on language mixing and translation.

        Args:
            predictions: Generated multilingual text
            references: Reference multilingual text

        Returns:
            Score between 0-1
        """
        if not predictions:
            return 0.0

        scores = []
        for pred in predictions:
            score = 0.0

            # Check for non-ASCII characters (indicator of non-English)
            if any(ord(c) > 127 for c in pred):
                score += 0.3

            # Check for translation markers
            if re.search(r'(translate|translation|in \w+ means|en français|en español)', pred.lower()):
                score += 0.2

            # Check for language names
            languages = ['english', 'spanish', 'french', 'german', 'chinese', 'japanese', 'arabic', 'russian']
            if any(lang in pred.lower() for lang in languages):
                score += 0.2

            # Check for common multilingual phrases
            phrases = ['hello', 'bonjour', 'hola', 'guten tag', 'ciao', 'こんにちは', 'مرحبا']
            if any(phrase in pred.lower() for phrase in phrases):
                score += 0.3

            scores.append(min(score, 1.0))

        return sum(scores) / len(scores)


class CreativeEvaluator:
    """Evaluate creative writing and generation."""

    def evaluate(self, predictions: List[str], references: List[str]) -> float:
        """
        Score creative content based on stylistic elements and variety.

        Args:
            predictions: Generated creative text
            references: Reference creative text

        Returns:
            Score between 0-1
        """
        if not predictions:
            return 0.0

        scores = []
        for pred in predictions:
            score = 0.0

            # Check for descriptive language
            descriptive_words = [
                'beautiful', 'mysterious', 'ancient', 'vibrant', 'serene', 'dramatic',
                'gentle', 'fierce', 'delicate', 'powerful', 'whisper', 'thunder',
            ]
            desc_count = sum(1 for word in descriptive_words if word.lower() in pred.lower())
            score += min(desc_count * 0.1, 0.3)

            # Check for metaphors/similes
            if re.search(r'(like a|as if|resembles|seems like)', pred.lower()):
                score += 0.2

            # Check for dialogue
            if '"' in pred or "'" in pred:
                score += 0.2

            # Check for varied sentence structure
            sentences = pred.split('.')
            if len(sentences) > 3:
                avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
                if 10 < avg_len < 25:  # Good variety
                    score += 0.15

            # Check for emotional language
            emotions = ['feel', 'felt', 'emotion', 'heart', 'soul', 'dream', 'hope', 'fear', 'love', 'hate']
            if any(emotion in pred.lower() for emotion in emotions):
                score += 0.15

            scores.append(min(score, 1.0))

        return sum(scores) / len(scores)


class FactualEvaluator:
    """Evaluate factual knowledge and accuracy."""

    def evaluate(self, predictions: List[str], references: List[str]) -> float:
        """
        Score factual content based on specificity and precision.

        Args:
            predictions: Generated factual text
            references: Reference factual text

        Returns:
            Score between 0-1
        """
        if not predictions:
            return 0.0

        scores = []
        for pred in predictions:
            score = 0.0

            # Check for specific dates
            if re.search(r'\b(19|20)\d{2}\b', pred):
                score += 0.2

            # Check for numbers/statistics
            if re.search(r'\d+%|\d+,\d+|\d+\.\d+', pred):
                score += 0.2

            # Check for proper nouns (capitalized words)
            words = pred.split()
            capitalized = sum(1 for w in words if w and w[0].isupper() and w not in ['I', 'The', 'A', 'An'])
            score += min(capitalized * 0.05, 0.3)

            # Check for citation-like patterns
            if re.search(r'(according to|source|study|research|report)', pred.lower()):
                score += 0.15

            # Check for precision markers
            if re.search(r'(exactly|precisely|specifically|particularly)', pred.lower()):
                score += 0.15

            scores.append(min(score, 1.0))

        return sum(scores) / len(scores)


class ConversationEvaluator:
    """Evaluate conversational and dialogue quality."""

    def evaluate(self, predictions: List[str], references: List[str]) -> float:
        """
        Score conversational quality based on engagement and flow.

        Args:
            predictions: Generated conversation text
            references: Reference conversation text

        Returns:
            Score between 0-1
        """
        if not predictions:
            return 0.0

        scores = []
        for pred in predictions:
            score = 0.0

            # Check for questions
            if '?' in pred:
                score += 0.2

            # Check for conversational markers
            markers = [
                'hello', 'hi', 'thanks', 'please', 'sorry', 'yes', 'no',
                'okay', 'sure', 'actually', 'well', 'so', 'anyway',
            ]
            marker_count = sum(1 for marker in markers if marker.lower() in pred.lower())
            score += min(marker_count * 0.1, 0.3)

            # Check for personal pronouns (more conversational)
            pronouns = ['I', 'you', 'we', 'my', 'your', 'our']
            pronoun_count = sum(pred.lower().count(f' {p.lower()} ') for p in pronouns)
            score += min(pronoun_count * 0.05, 0.2)

            # Check for politeness
            polite_words = ['please', 'thank', 'appreciate', 'kindly', 'grateful']
            if any(word in pred.lower() for word in polite_words):
                score += 0.15

            # Check for turn-taking indicators
            if re.search(r'(what do you think|your thoughts|tell me|let me know)', pred.lower()):
                score += 0.15

            scores.append(min(score, 1.0))

        return sum(scores) / len(scores)


# Domain evaluator registry
DOMAIN_EVALUATORS = {
    "code": CodeEvaluator(),
    "science": ScienceEvaluator(),
    "reasoning": ReasoningEvaluator(),
    "vision": VisionEvaluator(),
    "robotics": RoboticsEvaluator(),
    "multilingual": MultilingualEvaluator(),
    "creative": CreativeEvaluator(),
    "factual": FactualEvaluator(),
    "conversation": ConversationEvaluator(),
}


def get_domain_evaluator(domain: str):
    """Get evaluator for a specific domain."""
    return DOMAIN_EVALUATORS.get(domain)
