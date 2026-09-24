import math
import re
from collections import Counter
from urllib.parse import urlparse
from tld_popularity import is_popular_tld


FEATURE_NAMES = [
    "url_length",
    "has_ip_address",
    "dot_count",
    "https_flag",
    "url_entropy",
    "token_count",
    "subdomain_count",
    "query_param_count",
    "tld_length",
    "path_length",
    "has_hyphen_in_domain",
    "number_of_digits",
    "tld_popularity",
    "suspicious_file_extension",
    "domain_name_length",
    "percentage_numeric_chars",
]


def calculate_entropy(text):
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0

    counts = Counter(text)
    length = len(text)

    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def extract_features(url):
    """Extract the 16 features expected by the trained XGBoost model."""

    parsed = urlparse(url)

    hostname = parsed.netloc.split("@")[-1].split(":")[0]
    path = parsed.path
    query = parsed.query

    # Detect IPv4 address
    has_ip_address = int(
        bool(
            re.fullmatch(
                r"(?:\d{1,3}\.){3}\d{1,3}",
                hostname
            )
        )
    )

    # Count URL tokens separated by common delimiters
    tokens = [
        token
        for token in re.split(r"[./?=&_\-:@]+", url)
        if token
    ]

    # Count query parameters
    query_param_count = (
        len([p for p in query.split("&") if p])
        if query
        else 0
    )

    # Number of subdomains
    hostname_parts = [
        part for part in hostname.split(".") if part
    ]

    subdomain_count = max(
        0,
        len(hostname_parts) - 2
    )

    # Top-level domain length
    tld = hostname_parts[-1] if hostname_parts else ""

    # Digits
    number_of_digits = sum(
        character.isdigit()
        for character in url
    )

    # Percentage of URL characters that are numeric
    percentage_numeric_chars = (
        number_of_digits / len(url)
        if url
        else 0.0
    )

    # Hyphen in domain
    has_hyphen_in_domain = int("-" in hostname)

    # Suspicious file extensions
    suspicious_extensions = (
        ".exe",
        ".scr",
        ".zip",
        ".rar",
        ".js",
        ".jar",
        ".bat",
        ".cmd",
        ".msi",
        ".dll",
    )

    suspicious_file_extension = int(
        any(
            path.lower().endswith(extension)
            for extension in suspicious_extensions
        )
    )

    # Approximation of domain-name length
    domain_name_length = len(hostname)

    features = {
        "url_length": len(url),
        "has_ip_address": has_ip_address,
        "dot_count": url.count("."),
        "https_flag": int(parsed.scheme.lower() == "https"),
        "url_entropy": calculate_entropy(url),
        "token_count": len(tokens),
        "subdomain_count": subdomain_count,
        "query_param_count": query_param_count,
        "tld_length": len(tld),
        "path_length": len(path),
        "has_hyphen_in_domain": has_hyphen_in_domain,
        "number_of_digits": number_of_digits,
        "tld_popularity": is_popular_tld(tld),
        "suspicious_file_extension": suspicious_file_extension,
        "domain_name_length": domain_name_length,
        "percentage_numeric_chars": percentage_numeric_chars,
    }

    return features


if __name__ == "__main__":
    test_url = "https://example.com/login"

    features = extract_features(test_url)

    print("URL:", test_url)
    print("\nExtracted Features:")

    for name in FEATURE_NAMES:
        print(f"{name}: {features[name]}")