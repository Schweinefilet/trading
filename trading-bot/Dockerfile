FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p logs trades backtest_results

# Environment variables (override at runtime)
ENV TRADING_MODE=paper
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "from config.settings import settings; print('OK')" || exit 1

# Run the bot
CMD ["python", "bot.py"]
