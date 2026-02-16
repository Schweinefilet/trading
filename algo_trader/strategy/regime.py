"""
Regime-based permission logic.
Centralizes decisions on whether a trade is allowed based on the market environment.
"""

def regime_allows_trade(regime: str, direction: str) -> bool:
    """
    Filter trades by regime.
    
    Args:
        regime: Current market regime ("BULLISH", "CAUTIOUS", "NEUTRAL", "BEARISH", "CRISIS")
        direction: "LONG" or "SHORT"
        
    Returns:
        True if trade is permitted, False otherwise.
    """
    if regime == "CRISIS":
        return False
        
    if direction == "LONG":
        # Week 4: Restore CAUTIOUS to allow trading in moderately volatile windows
        return regime in ("BULLISH", "CAUTIOUS", "NEUTRAL")
        
    if direction == "SHORT":
        # No shorting for now, but placeholder for BEARISH
        return False
        
    return False
