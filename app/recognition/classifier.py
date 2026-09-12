"""
classifier.py - Content Classification Module for AI Smart Blackboard

This module provides rule-based content classification for text extracted
from the whiteboard. It identifies whether the content is mathematics,
physics, chemistry, general text, or diagrams using pattern matching and
keyword analysis.
"""

import re
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, field


class ContentCategory(Enum):
    """Enumeration of possible content categories."""
    MATHEMATICS = "MATHEMATICS"
    PHYSICS = "PHYSICS"
    CHEMISTRY = "CHEMISTRY"
    GENERAL_TEXT = "GENERAL_TEXT"
    DIAGRAM = "DIAGRAM"
    UNKNOWN = "UNKNOWN"


@dataclass
class ClassificationResult:
    """Container for classification results."""
    category: ContentCategory
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


class ContentClassifier:
    """
    Rule-based content classifier for whiteboard text.
    
    This classifier analyzes text using pattern matching and keyword detection
    to determine the content category. It is designed to be easily replaceable
    with ML-based classifiers in the future.
    
    Attributes:
        math_patterns: Regular expressions for mathematical content
        physics_keywords: Keywords indicating physics content
        chemistry_keywords: Keywords indicating chemistry content
        chemistry_formula_patterns: Patterns for chemical formulas
    """
    
    def __init__(self) -> None:
        """Initialize the ContentClassifier with classification rules."""
        # Mathematics patterns
        self._math_patterns = {
            'operators': re.compile(r'[+\-*/×÷=≠≈<>≤≥]'),
            'exponents': re.compile(r'[\d²³⁴⁵⁶⁷⁸⁹]|\^[0-9]'),
            'sqrt': re.compile(r'√'),
            'pi': re.compile(r'π'),
            'sigma': re.compile(r'Σ'),
            'integral': re.compile(r'∫'),
            'fraction': re.compile(r'\d+[/⁄]\d+'),
            'equation': re.compile(r'.+[=≠≈<>≤≥].+'),
            'variable': re.compile(r'[a-zA-Z](?=[+\-*/×÷=≠≈<>≤≥])'),
            'quadratic': re.compile(r'[a-zA-Z]²|\^2'),
            'function': re.compile(r'[a-zA-Z]\([^)]+\)'),
        }
        
        # Physics keywords
        self._physics_keywords = {
            'force', 'velocity', 'acceleration', 'energy', 'power',
            'momentum', 'gravity', 'newton', 'voltage', 'current',
            'resistance', 'magnetic', 'electric', 'motion', 'work',
            'wave', 'frequency', 'wavelength', 'mass', 'weight',
            'density', 'pressure', 'torque', 'inertia', 'friction',
            'kinetic', 'potential', 'thermal', 'mechanical', 'quantum',
            'relativity', 'photon', 'electron', 'proton', 'neutron',
            'atomic', 'nuclear', 'oscillation', 'amplitude', 'period',
            'hertz', 'joule', 'watt', 'volt', 'ohm', 'farad', 'henry',
            'tesla', 'weber', 'lumen', 'lux', 'pascal', 'bar',
        }
        
        # Chemistry keywords
        self._chemistry_keywords = {
            'acid', 'base', 'salt', 'water', 'hydrogen', 'oxygen',
            'nitrogen', 'carbon', 'sodium', 'chlorine', 'potassium',
            'calcium', 'magnesium', 'iron', 'copper', 'zinc', 'lead',
            'silver', 'gold', 'platinum', 'mercury', 'sulfur', 'phosphorus',
            'molecule', 'compound', 'element', 'atom', 'bond', 'ionic',
            'covalent', 'metallic', 'reaction', 'catalyst', 'enzyme',
            'ph', 'redox', 'oxidation', 'reduction', 'precipitate',
            'titration', 'distillation', 'filtration', 'chromatography',
        }
        
        # Chemical formula patterns
        self._chemistry_formula_patterns = [
            re.compile(r'[A-Z][a-z]?[A-Z]?[a-z]?[0-9]*'),  # Simple formula
            re.compile(r'[A-Z][a-z]?[0-9]*[A-Z]?[a-z]?[0-9]*'),  # Extended
            re.compile(r'[A-Z][a-z]?[0-9]*\([^)]+\)'),  # With parentheses
            re.compile(r'[A-Z][a-z]?[0-9]*[+\\-]'),  # With charge
            re.compile(r'[A-Z][a-z]?[0-9]*\s*[+\\-]'),  # With ion
        ]
        
        # Reaction arrow patterns
        self._reaction_patterns = [
            re.compile(r'→'),
            re.compile(r'->'),
            re.compile(r'⇌'),
            re.compile(r'<=>'),
            re.compile(r'→'),
            re.compile(r'←'),
        ]
        
    def preprocess_text(self, text: Optional[str]) -> str:
        """
        Clean and normalize the input text for classification.
        
        Args:
            text: Raw text string to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if text is None:
            return ""
        
        # Convert to string and strip whitespace
        processed = str(text).strip()
        
        # Remove excessive whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        return processed
        
    def detect_mathematics(self, text: str) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect if text contains mathematical content.
        
        Args:
            text: Preprocessed text string
            
        Returns:
            Tuple[bool, float, Dict]: (is_math, confidence, details)
        """
        details = {}
        score = 0.0
        total_weight = 0.0
        
        # Check for equation pattern (highest weight)
        if self._math_patterns['equation'].search(text):
            score += 30.0
            details['has_equation'] = True
            total_weight += 30.0
        
        # Check for operators
        operators = self._math_patterns['operators'].findall(text)
        if operators:
            score += min(len(operators) * 5, 25.0)
            details['operator_count'] = len(operators)
            total_weight += 25.0
        
        # Check for variables
        variables = self._math_patterns['variable'].findall(text)
        if variables:
            score += min(len(variables) * 3, 15.0)
            details['variable_count'] = len(variables)
            total_weight += 15.0
        
        # Check for exponents
        if self._math_patterns['exponents'].search(text):
            score += 15.0
            details['has_exponents'] = True
            total_weight += 15.0
        
        # Check for special symbols
        special_symbols = ['sqrt', 'pi', 'sigma', 'integral']
        for symbol in special_symbols:
            if self._math_patterns[symbol].search(text):
                score += 10.0
                details[f'has_{symbol}'] = True
                total_weight += 10.0
        
        # Check for fractions
        if self._math_patterns['fraction'].search(text):
            score += 10.0
            details['has_fraction'] = True
            total_weight += 10.0
        
        # Check for function notation
        if self._math_patterns['function'].search(text):
            score += 5.0
            details['has_function'] = True
            total_weight += 5.0
        
        # Normalize score
        confidence = min(score / max(total_weight, 1.0), 1.0)
        
        is_math = confidence >= 0.3
        
        return is_math, confidence, details
        
    def detect_physics(self, text: str) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect if text contains physics content.
        
        Args:
            text: Preprocessed text string
            
        Returns:
            Tuple[bool, float, Dict]: (is_physics, confidence, details)
        """
        details = {}
        text_lower = text.lower()
        words = text_lower.split()
        
        found_keywords = []
        for keyword in self._physics_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        details['found_keywords'] = found_keywords
        details['keyword_count'] = len(found_keywords)
        
        # Calculate confidence based on keyword density
        if len(words) > 0:
            keyword_density = len(found_keywords) / len(words)
            confidence = min(keyword_density * 5, 1.0)  # Scale for better sensitivity
        else:
            confidence = 0.0
        
        # Boost confidence for multiple keywords
        if len(found_keywords) >= 3:
            confidence = min(confidence + 0.3, 1.0)
        elif len(found_keywords) >= 2:
            confidence = min(confidence + 0.15, 1.0)
        
        is_physics = confidence >= 0.25
        
        return is_physics, confidence, details
        
    def detect_chemistry(self, text: str) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect if text contains chemistry content.
        
        Args:
            text: Preprocessed text string
            
        Returns:
            Tuple[bool, float, Dict]: (is_chemistry, confidence, details)
        """
        details = {}
        text_lower = text.lower()
        words = text_lower.split()
        score = 0.0
        
        # Check for chemical formulas (high weight)
        formula_found = False
        for pattern in self._chemistry_formula_patterns:
            matches = pattern.findall(text)
            if matches:
                # Filter out common words that might match
                filtered = [m for m in matches if len(m) > 1 and not m.islower()]
                if filtered:
                    formula_found = True
                    score += min(len(filtered) * 10, 30.0)
                    details['formula_count'] = len(filtered)
                    break
        
        if formula_found:
            details['has_formula'] = True
        
        # Check for reaction arrows
        arrow_found = False
        for pattern in self._reaction_patterns:
            if pattern.search(text):
                arrow_found = True
                score += 20.0
                details['has_reaction_arrow'] = True
                break
        
        # Check for chemistry keywords
        found_keywords = []
        for keyword in self._chemistry_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        details['found_keywords'] = found_keywords
        details['keyword_count'] = len(found_keywords)
        score += min(len(found_keywords) * 3, 20.0)
        
        # Check for element symbols (capital letters followed by lowercase)
        element_pattern = re.compile(r'[A-Z][a-z]?[0-9]*')
        elements = element_pattern.findall(text)
        # Filter out common words that are also element symbols
        common_words = {'I', 'A', 'V', 'In', 'As', 'At', 'Be', 'Co', 'Li', 'Ne'}
        filtered_elements = [e for e in elements if len(e) > 1 and e not in common_words]
        
        if len(filtered_elements) >= 3:
            score += 15.0
            details['element_symbols'] = filtered_elements[:5]
        
        # Calculate confidence
        total_possible = 70.0
        confidence = min(score / total_possible, 1.0)
        
        # Boost confidence if multiple indicators are present
        indicators = sum([
            bool(details.get('has_formula', False)),
            bool(details.get('has_reaction_arrow', False)),
            len(details.get('found_keywords', [])) >= 2,
            len(filtered_elements) >= 3
        ])
        
        if indicators >= 3:
            confidence = min(confidence + 0.25, 1.0)
        elif indicators >= 2:
            confidence = min(confidence + 0.15, 1.0)
        
        is_chemistry = confidence >= 0.25
        
        return is_chemistry, confidence, details
        
    def detect_general_text(self, text: str) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect if text contains general/plain text content.
        
        Args:
            text: Preprocessed text string
            
        Returns:
            Tuple[bool, float, Dict]: (is_general, confidence, details)
        """
        details = {}
        
        if not text or len(text) < 3:
            details['reason'] = 'text_too_short'
            return False, 0.0, details
        
        words = text.split()
        
        # Check for sentence structure (capitalization, punctuation)
        has_period = '.' in text
        has_question = '?' in text
        has_exclamation = '!' in text
        has_comma = ',' in text
        
        details['has_punctuation'] = {
            'period': has_period,
            'question': has_question,
            'exclamation': has_exclamation,
            'comma': has_comma
        }
        
        # Check for common English articles and words
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                       'can', 'had', 'her', 'was', 'one', 'our', 'out', 'who',
                       'will', 'with', 'have', 'from', 'this', 'they', 'what',
                       'when', 'that', 'which', 'would', 'could', 'should'}
        
        word_count = len(words)
        common_count = sum(1 for word in words if word.lower() in common_words)
        
        details['common_word_count'] = common_count
        details['total_words'] = word_count
        
        # Calculate confidence
        score = 0.0
        
        # Sentence structure indicators
        if has_period or has_question or has_exclamation:
            score += 20.0
        
        if has_comma:
            score += 5.0
        
        # Common words indicators
        if common_count > 0 and word_count > 0:
            common_ratio = common_count / word_count
            score += min(common_ratio * 30, 30.0)
        
        # Word count indicators
        if word_count >= 5:
            score += min(word_count * 2, 20.0)
        elif word_count >= 3:
            score += 10.0
        
        # Check if it reads like a sentence (has verb-like patterns)
        if re.search(r'\b(is|are|was|were|have|has|had|do|does|did|will|would|could|should|may|might|must)\b', text.lower()):
            score += 15.0
        
        total_possible = 90.0
        confidence = min(score / total_possible, 1.0)
        
        is_general = confidence >= 0.3 and word_count >= 3
        
        return is_general, confidence, details
        
    def detect_diagram(self, text: str) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect if text likely comes from a diagram.
        
        Args:
            text: Preprocessed text string
            
        Returns:
            Tuple[bool, float, Dict]: (is_diagram, confidence, details)
        """
        details = {}
        
        if not text or len(text.strip()) < 2:
            details['reason'] = 'empty_text'
            return True, 0.8, details
        
        # Count words and characters
        words = text.split()
        char_count = len(text.strip())
        word_count = len(words)
        
        details['char_count'] = char_count
        details['word_count'] = word_count
        
        # Diagrams typically have very short text
        if word_count <= 2 and char_count <= 20:
            # Check if it's a single label or term
            if len(words) == 1:
                # Check for uppercase acronyms (likely diagram labels)
                if text.isupper() and len(text) <= 5:
                    details['reason'] = 'uppercase_acronym'
                    return True, 0.85, details
                
                # Check for scientific notation or single term
                if re.search(r'[A-Z][a-z]?[0-9]*', text):
                    details['reason'] = 'scientific_label'
                    return True, 0.7, details
            
            # Check for axis labels (X, Y, etc.)
            if text in ['X', 'Y', 'Z', 'x', 'y', 'z']:
                details['reason'] = 'axis_label'
                return True, 0.9, details
            
            # Check for measurement units
            if re.search(r'\d+(?:\.\d+)?\s*(?:cm|m|km|g|kg|s|ms|V|A|W|Hz)', text):
                details['reason'] = 'measurement_unit'
                return True, 0.75, details
            
            # Generic single-term diagram label
            details['reason'] = 'single_term'
            return True, 0.6, details
        
        # Check for multiple short terms (often diagram labels)
        if word_count <= 5 and all(len(word) <= 8 for word in words):
            # Check if all words are captialized or short acronyms
            caps_count = sum(1 for word in words if word.isupper())
            if caps_count == word_count:
                details['reason'] = 'multiple_acronyms'
                return True, 0.75, details
            elif caps_count >= word_count * 0.5:
                details['reason'] = 'mixed_labels'
                return True, 0.6, details
        
        return False, 0.0, details
        
    def classify(self, text: Optional[str]) -> Dict[str, Any]:
        """
        Classify the input text into a content category.
        
        Args:
            text: Raw text string to classify
            
        Returns:
            Dict[str, Any]: Classification results containing:
                - category: The identified category (str)
                - confidence: Confidence score (float)
                - details: Additional classification details (dict)
        """
        try:
            # Preprocess the text
            cleaned_text = self.preprocess_text(text)
            
            # Handle empty text
            if not cleaned_text:
                return {
                    "category": ContentCategory.UNKNOWN.value,
                    "confidence": 0.0,
                    "details": {
                        "reason": "empty_text",
                        "message": "No text provided for classification"
                    }
                }
            
            # Store results from all detectors
            results = {}
            
            # Check for diagram first (it's often short text)
            is_diagram, diag_conf, diag_details = self.detect_diagram(cleaned_text)
            if is_diagram:
                results[ContentCategory.DIAGRAM] = diag_conf
                results['diagram_details'] = diag_details
            
            # Check for mathematics
            is_math, math_conf, math_details = self.detect_mathematics(cleaned_text)
            if is_math:
                results[ContentCategory.MATHEMATICS] = math_conf
                results['math_details'] = math_details
            
            # Check for chemistry
            is_chem, chem_conf, chem_details = self.detect_chemistry(cleaned_text)
            if is_chem:
                results[ContentCategory.CHEMISTRY] = chem_conf
                results['chem_details'] = chem_details
            
            # Check for physics
            is_phys, phys_conf, phys_details = self.detect_physics(cleaned_text)
            if is_phys:
                results[ContentCategory.PHYSICS] = phys_conf
                results['phys_details'] = phys_details
            
            # Check for general text
            is_gen, gen_conf, gen_details = self.detect_general_text(cleaned_text)
            if is_gen:
                results[ContentCategory.GENERAL_TEXT] = gen_conf
                results['gen_details'] = gen_details
            
            # If no specific category found
            if not results:
                # If there's text but no category, classify as GENERAL_TEXT
                if len(cleaned_text.split()) >= 3:
                    return {
                        "category": ContentCategory.GENERAL_TEXT.value,
                        "confidence": 0.3,
                        "details": {
                            "reason": "default_general",
                            "text_length": len(cleaned_text)
                        }
                    }
                else:
                    return {
                        "category": ContentCategory.UNKNOWN.value,
                        "confidence": 0.1,
                        "details": {
                            "reason": "no_category_matched",
                            "text_length": len(cleaned_text)
                        }
                    }
            
            # Sort categories by confidence and select the best
            category_results = {k: v for k, v in results.items() if isinstance(v, float)}
            
            if not category_results:
                return {
                    "category": ContentCategory.UNKNOWN.value,
                    "confidence": 0.0,
                    "details": {"reason": "no_valid_results"}
                }
            
            best_category = max(category_results, key=category_results.get)
            best_confidence = category_results[best_category]
            
            # Prepare response
            response = {
                "category": best_category.value,
                "confidence": best_confidence,
                "details": {
                    "all_results": {k.value if isinstance(k, ContentCategory) else k: v 
                                  for k, v in results.items() if isinstance(v, float)},
                    "text_preview": cleaned_text[:100] + ("..." if len(cleaned_text) > 100 else ""),
                    "text_length": len(cleaned_text),
                    "word_count": len(cleaned_text.split())
                }
            }
            
            return response
            
        except Exception as e:
            # Handle any unexpected errors
            return {
                "category": ContentCategory.UNKNOWN.value,
                "confidence": 0.0,
                "details": {
                    "error": str(e),
                    "message": "Classification failed due to an unexpected error"
                }
            }