POPULAR_TLDS = {
    "com",
    "org",
    "net",
    "edu",
    "gov",
    "mil",
    "int",
    "co",
    "io",
    "ai",
    "me",
    "info",
    "biz",
    "in",
    "uk",
    "de",
    "fr",
    "au",
    "ca",
    "us",
}


def is_popular_tld(tld):
    """Return 1 when the TLD is in our defined popular-TLD list."""
    return int(tld.lower() in POPULAR_TLDS)