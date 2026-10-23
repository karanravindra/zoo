import re

# Most frequent English function words. Prose in other Latin-script languages
# (Spanish, French, German, ...) uses almost none of them.
STOPWORDS = frozenset(
    ["the", "of", "and", "to", "a", "in", "is", "that", "for", "it", "with", "as", "was", "on", "are", "be", "this", "by", "an", "at", "or", "from", "you", "i", "have", "not", "but", "they", "his", "he", "we", "which", "their", "has", "were", "will", "can", "all", "there", "been", "one", "would", "more", "when", "who", "what", "so", "if", "about", "into", "than", "them", "its"]
)
WORD = re.compile(r"[A-Za-z']+")
COMMENT = re.compile(r"#\s*(.+)|\"\"\"(.+?)\"\"\"", re.DOTALL)

SAMPLE_CHARS = 3000


def latin_letter_share(text):
    """Share of letters that are plain ASCII, which rejects CJK, Cyrillic, Arabic, ..."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(c < "\x80" for c in letters) / len(letters)


def stopword_share(text, min_words):
    words = WORD.findall(text)
    if len(words) < min_words:
        return None
    return sum(w.lower() in STOPWORDS for w in words) / len(words)


def is_english_prose(text, min_latin=0.95, min_stopwords=0.10):
    sample = text[:SAMPLE_CHARS]
    if latin_letter_share(sample) < min_latin:
        return False
    share = stopword_share(sample, min_words=25)
    return share is None or share >= min_stopwords


def is_english_code(text, min_latin=0.98, min_stopwords=0.04):
    """Check the script of the whole file, and the language of comments when there is enough prose.

    English code comments are terse, so the stopword bar is low and only applies to
    files with plenty of comment text.
    """
    if latin_letter_share(text) < min_latin:
        return False
    comments = " ".join(a or b for a, b in COMMENT.findall(text[:20_000]))
    share = stopword_share(comments, min_words=60)
    return share is None or share >= min_stopwords


def is_english_chat(text, min_latin=0.95):
    return latin_letter_share(text) >= min_latin


CHECKS = {"prose": is_english_prose, "code": is_english_code, "chat": is_english_chat}
