"""
Text Data Cleaning and Filtering for LLM Pretraining
This script cleans and filters text data to exclude junk while keeping
content related to code, tech, science, bio, QA, etc.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
import hashlib
from collections import defaultdict

class TextCleaner:
    """Handles text cleaning operations"""

    def __init__(self):
        # Common patterns to clean
        self.url_pattern = re.compile(r'https?://\S+')
        self.email_pattern = re.compile(r'\S+@\S+\.\S+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.multiple_spaces = re.compile(r'\s+')
        self.multiple_newlines = re.compile(r'\n{3,}')

    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        return unicodedata.normalize('NFKC', text)

    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags"""
        return self.html_pattern.sub('', text)

    def clean_urls(self, text: str, keep_urls: bool = True) -> str:
        """Clean or remove URLs"""
        if keep_urls:
            # Replace long URLs with placeholder
            return re.sub(r'https?://[^\s]{100,}', '[URL]', text)
        else:
            return self.url_pattern.sub('', text)

    def clean_whitespace(self, text: str) -> str:
        """Clean excessive whitespace"""
        text = self.multiple_spaces.sub(' ', text)
        text = self.multiple_newlines.sub('\n\n', text)
        return text.strip()

    def remove_control_characters(self, text: str) -> str:
        """Remove control characters except newlines and tabs"""
        return ''.join(char for char in text
                      if unicodedata.category(char)[0] != 'C'
                      or char in '\n\t')

    def clean_text(self, text: str, keep_urls: bool = True) -> str:
        """Apply all cleaning steps"""
        if not text:
            return ""

        text = self.normalize_unicode(text)
        text = self.remove_html_tags(text)
        text = self.clean_urls(text, keep_urls)
        text = self.remove_control_characters(text)
        text = self.clean_whitespace(text)

        return text


class TextFilter:
    """Handles text filtering based on quality criteria"""

    def __init__(self, config: Dict):
        self.min_length = config.get('min_length', 100)
        self.max_length = config.get('max_length', 1000000)
        self.keep_keywords = [kw.lower() for kw in config.get('keep_keywords', [])]
        self.exclude_keywords = [kw.lower() for kw in config.get('exclude_keywords', [])]
        self.junk_patterns = [re.compile(pattern) for pattern in config.get('junk_patterns', [])]

    def check_length(self, text: str) -> bool:
        """Check if text length is within acceptable range"""
        length = len(text)
        return self.min_length <= length <= self.max_length

    def check_junk_patterns(self, text: str) -> bool:
        """Check for junk patterns - returns True if text is clean"""
        for pattern in self.junk_patterns:
            if pattern.search(text):
                return False
        return True

    def check_exclude_keywords(self, text: str) -> bool:
        """Check for exclude keywords - returns True if no exclude keywords found"""
        text_lower = text.lower()
        for keyword in self.exclude_keywords:
            if keyword in text_lower:
                return False
        return True

    def calculate_relevance_score(self, text: str) -> float:
        """Calculate relevance score based on keep keywords"""
        if not self.keep_keywords:
            return 1.0  # If no keywords specified, accept all

        text_lower = text.lower()
        matches = sum(1 for keyword in self.keep_keywords if keyword in text_lower)
        return matches / len(self.keep_keywords)

    def has_code_indicators(self, text: str) -> bool:
        """Check if text contains code-like patterns"""
        code_patterns = [
            r'\bdef\s+\w+\s*\(',  # Python function
            r'\bfunction\s+\w+\s*\(',  # JavaScript function
            r'\bclass\s+\w+\s*[:{]',  # Class definition
            r'\bpublic\s+\w+\s+\w+\s*\(',  # Java/C# method
            r'import\s+[\w.]+',  # Import statement
            r'#include\s*<[\w.]+>',  # C/C++ include
            r'```[\w]*\n',  # Markdown code block
            r'(?:int|string|bool|float|double)\s+\w+\s*=',  # Variable declaration
        ]

        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        return False

    def is_valid(self, text: str, min_relevance: float = 0.0) -> Tuple[bool, Dict]:
        """
        Check if text passes all filters
        Returns: (is_valid, details_dict)
        """
        details = {
            'length_ok': False,
            'no_junk_patterns': False,
            'no_exclude_keywords': False,
            'relevance_score': 0.0,
            'has_code': False
        }

        # Length check
        details['length_ok'] = self.check_length(text)
        if not details['length_ok']:
            return False, details

        # Junk patterns check
        details['no_junk_patterns'] = self.check_junk_patterns(text)
        if not details['no_junk_patterns']:
            return False, details

        # Exclude keywords check
        details['no_exclude_keywords'] = self.check_exclude_keywords(text)
        if not details['no_exclude_keywords']:
            return False, details

        # Relevance score
        details['relevance_score'] = self.calculate_relevance_score(text)

        # Code indicators
        details['has_code'] = self.has_code_indicators(text)

        # Accept if relevance score is above threshold OR if it has code
        is_valid = (details['relevance_score'] >= min_relevance) or details['has_code']

        return is_valid, details


class TextDeduplicator:
    """Handles text deduplication"""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes = set()
        self.seen_texts = []

    def get_hash(self, text: str) -> str:
        """Get hash of text for exact deduplication"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_shingles(self, text: str, k: int = 5) -> set:
        """Get character shingles for fuzzy deduplication"""
        words = text.split()
        if len(words) < k:
            return {text}
        return {' '.join(words[i:i+k]) for i in range(len(words) - k + 1)}

    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def is_duplicate(self, text: str, use_fuzzy: bool = True) -> bool:
        """Check if text is a duplicate"""
        # Exact deduplication
        text_hash = self.get_hash(text)
        if text_hash in self.seen_hashes:
            return True

        # Fuzzy deduplication (disabled by default for performance)
        if use_fuzzy and len(self.seen_texts) < 10000:  # Limit for performance
            text_shingles = self.get_shingles(text)
            for seen_text in self.seen_texts[-1000:]:  # Check last 1000 only
                seen_shingles = self.get_shingles(seen_text)
                similarity = self.jaccard_similarity(text_shingles, seen_shingles)
                if similarity >= self.similarity_threshold:
                    return True

        # Add to seen
        self.seen_hashes.add(text_hash)
        if use_fuzzy and len(self.seen_texts) < 10000:
            self.seen_texts.append(text)

        return False


class DatasetCleaner:
    """Main class for cleaning and filtering datasets"""

    def __init__(self, config: Dict):
        self.cleaner = TextCleaner()
        self.filter = TextFilter(config.get('filtering', {}))

        dedup_config = config.get('filtering', {}).get('deduplication', {})
        self.use_deduplication = dedup_config.get('enabled', True)
        self.deduplicator = TextDeduplicator(
            similarity_threshold=dedup_config.get('similarity_threshold', 0.85)
        ) if self.use_deduplication else None

        self.stats = defaultdict(int)

    def process_text(self, text: str, min_relevance: float = 0.0) -> Optional[str]:
        """
        Process a single text: clean, filter, and deduplicate
        Returns: cleaned text if valid, None otherwise
        """
        self.stats['total'] += 1

        if not text or not isinstance(text, str):
            self.stats['empty'] += 1
            return None

        # Clean text
        cleaned = self.cleaner.clean_text(text)

        if not cleaned:
            self.stats['empty_after_cleaning'] += 1
            return None

        # Filter text
        is_valid, details = self.filter.is_valid(cleaned, min_relevance)

        if not is_valid:
            if not details['length_ok']:
                self.stats['filtered_length'] += 1
            elif not details['no_junk_patterns']:
                self.stats['filtered_junk'] += 1
            elif not details['no_exclude_keywords']:
                self.stats['filtered_keywords'] += 1
            else:
                self.stats['filtered_relevance'] += 1
            return None

        # Deduplicate
        if self.use_deduplication and self.deduplicator:
            if self.deduplicator.is_duplicate(cleaned, use_fuzzy=False):
                self.stats['duplicate'] += 1
                return None

        # Track kept text
        self.stats['kept'] += 1
        if details['has_code']:
            self.stats['kept_with_code'] += 1

        return cleaned

    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return dict(self.stats)

    def print_stats(self):
        """Print processing statistics"""
        print("\n=== Processing Statistics ===")
        print(f"Total texts processed: {self.stats['total']}")
        print(f"Kept: {self.stats['kept']} ({self.stats['kept']/max(1, self.stats['total'])*100:.2f}%)")
        print(f"  - With code: {self.stats['kept_with_code']}")
        print(f"Filtered out: {self.stats['total'] - self.stats['kept']}")
        print(f"  - Empty: {self.stats['empty']}")
        print(f"  - Empty after cleaning: {self.stats['empty_after_cleaning']}")
        print(f"  - Length: {self.stats['filtered_length']}")
        print(f"  - Junk patterns: {self.stats['filtered_junk']}")
        print(f"  - Exclude keywords: {self.stats['filtered_keywords']}")
        print(f"  - Low relevance: {self.stats['filtered_relevance']}")
        print(f"  - Duplicates: {self.stats['duplicate']}")
        print("=" * 40)


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load config
    with open('datasets_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create cleaner
    cleaner = DatasetCleaner(config)

    # Test examples
    test_texts = [
        "This is a simple Python function:\ndef hello():\n    print('Hello World')",
        "BUY NOW!!! LIMITED OFFER!!! Click here to win $$$$$",
        "Research shows that machine learning algorithms can improve performance",
        "a",  # Too short
        "Subscribe now to our casino newsletter!!!!! WINNER WINNER"
    ]

    print("Testing text cleaning and filtering:\n")
    for i, text in enumerate(test_texts, 1):
        result = cleaner.process_text(text)
        print(f"Text {i}:")
        print(f"  Original: {text[:80]}...")
        print(f"  Result: {'KEPT' if result else 'FILTERED'}")
        if result:
            print(f"  Cleaned: {result[:80]}...")
        print()

    cleaner.print_stats()
