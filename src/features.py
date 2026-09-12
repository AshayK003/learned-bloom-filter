"""URL feature extraction for learned bloom filters.

Feature groups:
  A: Character n-grams (TF-IDF)
  B: Structural features (lengths, counts, ratios)
  C: Statistical features (entropy)
"""

import math
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from urllib.parse import urlparse


class StructuralFeatures(BaseEstimator, TransformerMixin):
    """Extract structural features from URLs."""

    def __init__(self):
        self.shorteners = {
            'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly',
            'is.gd', 'buff.ly', 'adf.ly', 'bit.do', 'soo.gd',
        }

    def fit(self, X, y=None):
        return self

    def transform(self, urls):
        features = np.array([self._extract(url) for url in urls], dtype=np.float64)
        return features

    def _extract(self, url):
        try:
            parsed = urlparse(url if '://' in url else f'http://{url}')
        except Exception:
            parsed = urlparse('http://invalid')

        hostname = parsed.hostname or ''
        path = parsed.path or ''
        query = parsed.query or ''
        full = url

        # Character counts
        num_digits = sum(c.isdigit() for c in full)
        num_alpha = sum(c.isalpha() for c in full)
        num_special = len(full) - num_digits - num_alpha

        # Domain tokens
        domain_tokens = re.split(r'[.\-_]', hostname)
        domain_tokens = [t for t in domain_tokens if t]
        longest_token = max((len(t) for t in domain_tokens), default=0)
        shortest_token = min((len(t) for t in domain_tokens), default=0)

        # Ratios
        total_len = len(full) if full else 1
        digit_ratio = num_digits / total_len
        special_ratio = num_special / total_len
        alpha_ratio = num_alpha / total_len

        # Entropy
        entropy = self._entropy(full)
        entropy_domain = self._entropy(hostname) if hostname else 0.0

        # Suspicious patterns
        has_ip = bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', hostname))
        has_port = parsed.port is not None and parsed.port not in (80, 443)
        has_encoding = '%' in full

        # TLD
        tld = hostname.rsplit('.', 1)[-1] if '.' in hostname else ''
        is_short = hostname in self.shorteners or any(
            hostname.endswith(s) for s in self.shorteners
        )

        return [
            len(full),                          # url_length
            len(hostname),                      # hostname_length
            len(path),                          # path_length
            len(query),                         # query_length
            full.count('.'),                    # num_dots
            full.count('-'),                    # num_hyphens
            full.count('_'),                    # num_underscores
            full.count('/'),                    # num_slashes
            full.count('@'),                    # num_at_signs
            full.count('&'),                    # num_ampersands
            full.count('='),                    # num_equals
            num_digits,                         # num_digits
            num_special,                        # num_special
            int(has_ip),                        # has_ip_address
            int(parsed.scheme == 'https'),      # has_https
            int(has_port),                      # has_port_in_url
            hostname.count('.') + 1 if hostname else 0,  # subdomain_depth
            path.count('/') if path else 0,     # path_depth
            longest_token,                      # longest_domain_token
            shortest_token,                     # shortest_domain_token
            int(has_encoding),                  # has_encoding
            int(bool(tld)),                     # has_tld
            int(is_short),                      # is_shortened
            digit_ratio,                        # digit_ratio
            special_ratio,                      # special_ratio
            alpha_ratio,                        # alpha_ratio
            entropy,                            # entropy
            entropy_domain,                     # entropy_domain
        ]

    def _entropy(self, s):
        """Shannon entropy of a string."""
        if not s:
            return 0.0
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        length = len(s)
        return -sum((count / length) * math.log2(count / length) for count in freq.values())

    def get_feature_names(self):
        return [
            'url_length', 'hostname_length', 'path_length', 'query_length',
            'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
            'num_at_signs', 'num_ampersands', 'num_equals', 'num_digits',
            'num_special', 'has_ip_address', 'has_https', 'has_port_in_url',
            'subdomain_depth', 'path_depth', 'longest_domain_token',
            'shortest_domain_token', 'has_encoding', 'has_tld', 'is_shortened',
            'digit_ratio', 'special_ratio', 'alpha_ratio', 'entropy',
            'entropy_domain',
        ]


def build_feature_extractor(ngram_range=(1, 5), max_features=5000, use_structural=True, use_statistical=True):
    """Build a FeatureUnion combining character n-grams + structural features.

    Args:
        ngram_range: Range of n-grams for character TF-IDF
        max_features: Maximum number of TF-IDF features
        use_structural: Whether to include structural features
        use_statistical: Whether to include statistical features (entropy, ratios)

    Returns:
        sklearn Pipeline or FeatureUnion
    """
    transformers = []

    # Character n-grams
    char_tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=True,
        strip_accents=None,
    )
    transformers.append(('char_tfidf', char_tfidf))

    # Structural features (includes statistical)
    struct = StructuralFeatures()
    transformers.append(('struct', struct))

    return FeatureUnion(transformer_list=transformers)
