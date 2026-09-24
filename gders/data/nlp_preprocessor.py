"""
GDERS NLP Preprocessor Module.

Implements the deterministic NLP preprocessing pipeline for the frozen GDERS corpus
according to the research methodology:
- URL normalization/removal
- GitHub mention (@user) and issue reference (#123) normalization
- Fenced code block and inline code extraction/normalization
- Special character stripping with technical identifier preservation
- Lowercasing and whitespace normalization
- Tokenization
- Standard stop-word removal (NLTK English stopwords)
- Deterministic lemmatization (NLTK WordNetLemmatizer with fallback)
- Quality diagnostics and traceability to raw review comments
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProcessedCommentRecord:
    """Standardized GDERS Processed Comment Data Contract with Complete Traceability."""
    # Provenance inherited from raw record
    repository: str
    owner: str
    repo: str
    repository_url: str
    pull_request_number: int
    pull_request_url: str
    pull_request_title: str
    pull_request_author: str
    comment_id: int
    commenter_login: str
    commenter_id: Optional[int]
    comment_url: str
    created_at: str
    updated_at: str
    commit_id: Optional[str] = None
    file_path: Optional[str] = None
    diff_hunk: Optional[str] = None
    collection_timestamp: Optional[str] = None
    collection_window_start: Optional[str] = None
    collection_window_end: Optional[str] = None

    # Immutable Raw Body Preservation
    original_body: str = ""

    # Phase 4 Preprocessed Outputs
    cleaned_text: str = ""
    processed_text: str = ""
    tokens: List[str] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)

    # Diagnostic & Quality Flags
    character_count_original: int = 0
    character_count_processed: int = 0
    token_count: int = 0
    unique_token_count: int = 0
    is_empty_after_preprocessing: bool = False
    has_code_block: bool = False
    has_inline_code: bool = False
    has_url: bool = False
    has_mention: bool = False
    has_issue_ref: bool = False
    preprocessing_status: str = "success"  # 'success', 'empty_after_preprocessing', 'malformed_input'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GDERSNLPPreprocessor:
    """Deterministic NLP Preprocessing Engine for GDERS review comments."""

    # Pre-compiled Regex Patterns for GitHub Review Comment Normalization
    URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    FENCED_CODE_REGEX = re.compile(r"```[\s\S]*?```")
    INLINE_CODE_REGEX = re.compile(r"`[^`]+`")
    MENTION_REGEX = re.compile(r"(?<!\w)@([a-zA-Z0-9_\-]+)")
    ISSUE_REF_REGEX = re.compile(r"(?<!\w)#(\d+)")
    
    # Preserve technical punctuation like hyphens inside tokens if alphanumeric
    PUNCT_REGEX = re.compile(r"[^\w\s\-]")
    MULTIPLE_SPACES_REGEX = re.compile(r"\s+")

    def __init__(self, custom_stopwords: Optional[Set[str]] = None):
        self._init_nlp_resources(custom_stopwords)

    def _init_nlp_resources(self, custom_stopwords: Optional[Set[str]] = None) -> None:
        """Initialize NLTK stopwords and WordNetLemmatizer."""
        self.lemmatizer_name = "WordNetLemmatizer"
        self.lemmatizer_version = getattr(nltk, "__version__", "unknown") if NLTK_AVAILABLE else "rule-based-fallback"
        self.stopwords_source = "NLTK English" if NLTK_AVAILABLE else "GDERS Default English"

        if NLTK_AVAILABLE:
            try:
                self.stopwords: Set[str] = set(nltk_stopwords.words("english"))
                self.lemmatizer = WordNetLemmatizer()
            except Exception as e:
                logger.warning(f"Could not load NLTK resources: {e}. Falling back to default list.")
                self.stopwords = set(self._get_default_stopwords())
                self.lemmatizer = None
        else:
            self.stopwords = set(self._get_default_stopwords())
            self.lemmatizer = None

        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

    @staticmethod
    def _get_default_stopwords() -> Set[str]:
        return {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
            "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
            "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
            "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
            "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
            "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
            "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
            "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
            "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
            "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
            "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
            "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
            "they've", "this", "those", "through", "to", "too", "under", "until", "up",
            "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
            "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
            "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
            "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
            "yourself", "yourselves",
        }

    def detect_features(self, text: str) -> Dict[str, bool]:
        """Detect presence of code blocks, URLs, mentions, and issue refs in raw text."""
        if not text:
            return {
                "has_code_block": False,
                "has_inline_code": False,
                "has_url": False,
                "has_mention": False,
                "has_issue_ref": False,
            }
        return {
            "has_code_block": bool(self.FENCED_CODE_REGEX.search(text)),
            "has_inline_code": bool(self.INLINE_CODE_REGEX.search(text)),
            "has_url": bool(self.URL_REGEX.search(text)),
            "has_mention": bool(self.MENTION_REGEX.search(text)),
            "has_issue_ref": bool(self.ISSUE_REF_REGEX.search(text)),
        }

    def clean_text(self, text: str) -> str:
        """Step-by-step deterministic text cleaning and normalization."""
        if not text:
            return ""

        # 1. URL removal
        cleaned = self.URL_REGEX.sub(" ", text)

        # 2. Fenced code block handling (remove large multi-line snippets)
        cleaned = self.FENCED_CODE_REGEX.sub(" ", cleaned)

        # 3. Inline code handling: extract identifier contents (strip the backticks)
        cleaned = self.INLINE_CODE_REGEX.sub(lambda m: " " + m.group(0).replace("`", "") + " ", cleaned)

        # 4. GitHub mentions & issue refs normalization
        cleaned = self.MENTION_REGEX.sub(" ", cleaned)
        cleaned = self.ISSUE_REF_REGEX.sub(" ", cleaned)

        # 5. Special character stripping (retain hyphens for compound technical words e.g. thread-safe)
        cleaned = self.PUNCT_REGEX.sub(" ", cleaned)

        # 6. Lowercasing
        cleaned = cleaned.lower()

        # 7. Whitespace normalization
        cleaned = self.MULTIPLE_SPACES_REGEX.sub(" ", cleaned).strip()

        return cleaned

    def tokenize(self, cleaned_text: str) -> List[str]:
        """Deterministic tokenization with stop-word removal and length bounds."""
        if not cleaned_text:
            return []

        raw_tokens = cleaned_text.split()
        tokens: List[str] = []

        for tok in raw_tokens:
            # Strip leading/trailing hyphens or underscores
            tok = tok.strip("-_")
            if not tok:
                continue

            # Length bound: preserve 2+ character words, or 1-char meaningful symbols if needed
            if len(tok) < 2:
                continue

            # Ignore pure numeric tokens
            if tok.isdigit():
                continue

            # Stopword filtering
            if tok in self.stopwords:
                continue

            tokens.append(tok)

        return tokens

    def lemmatize(self, token: str) -> str:
        """Deterministic lemmatization using WordNetLemmatizer with fallback."""
        if not token:
            return ""
        if self.lemmatizer:
            try:
                # Primary lemmatization as noun, then verb
                lemma_n = self.lemmatizer.lemmatize(token, pos="n")
                if lemma_n != token:
                    return lemma_n
                return self.lemmatizer.lemmatize(token, pos="v")
            except Exception:
                pass

        # Deterministic rule-based fallback if WordNet is unavailable
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ing") and len(token) > 5:
            return token[:-3]
        if token.endswith("ed") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            return token[:-1]
        return token

    def process_record(self, raw_record: Dict[str, Any]) -> ProcessedCommentRecord:
        """Process a single raw review comment dictionary and produce a ProcessedCommentRecord."""
        raw_body = raw_record.get("comment_body", "") or ""
        features = self.detect_features(raw_body)
        cleaned = self.clean_text(raw_body)
        tokens = self.tokenize(cleaned)
        lemmas = [self.lemmatize(t) for t in tokens]
        processed_text = " ".join(lemmas)

        char_orig = len(raw_body)
        char_proc = len(processed_text)
        tok_count = len(lemmas)
        unique_tok_count = len(set(lemmas))
        is_empty = tok_count == 0

        status = "empty_after_preprocessing" if is_empty else "success"

        return ProcessedCommentRecord(
            repository=raw_record.get("repository", ""),
            owner=raw_record.get("owner", ""),
            repo=raw_record.get("repo", ""),
            repository_url=raw_record.get("repository_url", ""),
            pull_request_number=raw_record.get("pull_request_number", 0),
            pull_request_url=raw_record.get("pull_request_url", ""),
            pull_request_title=raw_record.get("pull_request_title", ""),
            pull_request_author=raw_record.get("pull_request_author", ""),
            comment_id=raw_record.get("comment_id", 0),
            commenter_login=raw_record.get("commenter_login", ""),
            commenter_id=raw_record.get("commenter_id"),
            comment_url=raw_record.get("comment_url", ""),
            created_at=raw_record.get("created_at", ""),
            updated_at=raw_record.get("updated_at", ""),
            commit_id=raw_record.get("commit_id"),
            file_path=raw_record.get("file_path"),
            diff_hunk=raw_record.get("diff_hunk"),
            collection_timestamp=raw_record.get("collection_timestamp"),
            collection_window_start=raw_record.get("collection_window_start"),
            collection_window_end=raw_record.get("collection_window_end"),
            original_body=raw_body,
            cleaned_text=cleaned,
            processed_text=processed_text,
            tokens=tokens,
            lemmas=lemmas,
            character_count_original=char_orig,
            character_count_processed=char_proc,
            token_count=tok_count,
            unique_token_count=unique_tok_count,
            is_empty_after_preprocessing=is_empty,
            has_code_block=features["has_code_block"],
            has_inline_code=features["has_inline_code"],
            has_url=features["has_url"],
            has_mention=features["has_mention"],
            has_issue_ref=features["has_issue_ref"],
            preprocessing_status=status,
        )

