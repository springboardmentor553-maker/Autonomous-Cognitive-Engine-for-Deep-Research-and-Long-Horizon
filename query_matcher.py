"""
Ambiguous Query Detector — Deduplication via Jaccard Similarity.
Compares incoming queries against existing conversations to avoid redundant research.
"""
import re
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# Common English stopwords to filter out before comparison
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may", "might", "shall",
    "can", "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "we", "our", "you", "your", "he", "she", "they", "them", "their", "what",
    "which", "who", "whom", "how", "when", "where", "why", "with", "from",
    "about", "into", "through", "during", "before", "after", "above", "below",
    "between", "not", "no", "nor", "but", "if", "then", "so", "very", "just",
    "than", "too", "also", "more", "most", "some", "any", "all", "each", "every",
    "tell", "explain", "describe", "analyze", "research", "find", "give", "make",
    "please", "want", "need", "know", "think", "like", "get", "let"
}


def normalize_query(text: str) -> set:
    """
    Normalize a query into a set of meaningful tokens.
    - Lowercase
    - Remove punctuation
    - Filter stopwords
    - Return as a set for Jaccard comparison
    """
    # Lowercase and remove non-alphanumeric (keep spaces)
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    tokens = cleaned.split()
    # Filter stopwords and short tokens (≤2 chars)
    meaningful = {t for t in tokens if t not in STOPWORDS and len(t) > 2}
    return meaningful


def compute_similarity(query_a: str, query_b: str) -> float:
    """
    Compute Jaccard similarity between two queries.
    Returns a float between 0.0 (no overlap) and 1.0 (identical).
    """
    tokens_a = normalize_query(query_a)
    tokens_b = normalize_query(query_b)

    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union) if union else 0.0


def find_matching_conversation(user_id: int, new_query: str, threshold: float = 0.6) -> Optional[Dict]:
    """
    Search existing conversations for a match above the similarity threshold.
    
    Args:
        user_id: The user's database ID
        new_query: The incoming research query
        threshold: Minimum similarity to consider a match (default: 0.6)
        
    Returns:
        Dict with {conversation_id, title, similarity} if match found, else None.
    """
    # Import here to avoid circular imports
    from database import Conversation, Message

    try:
        # Get all conversations for this user
        conversations = Conversation.query.filter_by(user_id=user_id).all()

        if not conversations:
            return None

        best_match = None
        best_score = 0.0

        for convo in conversations:
            # Compare against conversation title
            title_sim = compute_similarity(new_query, convo.title or "")

            # Also compare against the first user message (more detailed)
            first_msg = Message.query.filter_by(
                conversation_id=convo.id, role='user'
            ).order_by(Message.created_at).first()

            msg_sim = 0.0
            if first_msg:
                msg_sim = compute_similarity(new_query, first_msg.content)

            # Use the higher of the two scores
            score = max(title_sim, msg_sim)

            if score > best_score:
                best_score = score
                best_match = {
                    "conversation_id": convo.id,
                    "title": convo.title,
                    "similarity": round(score, 3)
                }

        if best_match and best_score >= threshold:
            logger.info(
                f"[QUERY MATCH] Found match: '{best_match['title']}' "
                f"(similarity: {best_score:.2f})"
            )
            return best_match

        logger.info(f"[QUERY MATCH] No match above threshold {threshold} "
                     f"(best: {best_score:.2f})")
        return None

    except Exception as e:
        logger.error(f"[QUERY MATCH] Error during matching: {e}")
        return None
