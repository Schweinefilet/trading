from datetime import datetime, timedelta
from models import StockCache, db

class CacheService:
    @staticmethod
    def get(ticker, data_type, key_params=None):
        """
        Retrieves cached data if it exists and is within its TTL.
        TTLs:
        - quote: 15 minutes
        - fundamentals: 24 hours
        - financials: 7 days
        - history: 1 hour (default)
        - correlation: 24 hours
        """
        cache_entry = StockCache.query.filter_by(
            ticker=ticker, 
            data_type=data_type,
            key_params=str(key_params) if key_params else None
        ).first()
        
        if not cache_entry:
            return None
            
        ttls = {
            'quote': timedelta(minutes=15),
            'analyst': timedelta(minutes=15),
            'fundamentals': timedelta(hours=24),
            'financials': timedelta(days=7),
            'history': timedelta(hours=1),
            'correlation': timedelta(hours=24)
        }
        
        ttl = ttls.get(data_type, timedelta(hours=1))
        
        if datetime.utcnow() - cache_entry.fetched_at > ttl:
            return None
            
        return cache_entry.data

    @staticmethod
    def set(ticker, data_type, data, key_params=None):
        """
        Stores or updates cached data.
        """
        cache_entry = StockCache.query.filter_by(
            ticker=ticker, 
            data_type=data_type,
            key_params=str(key_params) if key_params else None
        ).first()
        
        if cache_entry:
            cache_entry.data = data
            cache_entry.fetched_at = datetime.utcnow()
        else:
            cache_entry = StockCache(
                ticker=ticker,
                data_type=data_type,
                key_params=str(key_params) if key_params else None,
                data=data
            )
            db.session.add(cache_entry)
        
        db.session.commit()
