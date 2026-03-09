# Stock Market Analytics & Portfolio Management App (Local)

A local full-stack stock market analytics and portfolio management application.

## Tech Stack

- **Backend:** Python + Flask
- **Frontend:** React (Vite) + Tailwind CSS + Recharts
- **Database:** SQLite (SQLAlchemy)
- **Data:** yfinance + Finnhub

## Project Structure

- `/backend`: Python Flask API and SQLite database
- `/frontend`: React client application

## Setup Instructions

### 1. Prerequisites

- Python 3.10+
- Node.js & npm

### 2. Backend Setup

1. Open a terminal in the `backend/` directory.
2. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the `backend/` directory:

   ```env
   FINNHUB_API_KEY=your_free_key_here
   ```

   *Get a free key at [finnhub.io](https://finnhub.io/)*

5. Start the backend server:

   ```bash
   python app.py
   ```

   The backend will run on `http://localhost:5001`.

### 3. Frontend Setup

1. Open a new terminal in the `frontend/` directory.
2. Install dependencies:

   ```bash
   npm install
   ```

3. Start the development server:

   ```bash
   npm run dev
   ```

4. Open your browser to the URL provided by Vite (usually `http://localhost:5173`).

---

## Features

- **Real-time Quotes**: Live price updates from Finnhub with yfinance fallback.
- **Advanced Charting**: Interactive charts with SMA, EMA, and Bollinger Band overlays.
- **Technical Indicators**: RSI, MACD, and Stochastic Oscillator panels.
- **Fundamental Deep Dive**: Core valuation and financial health metrics.
- **Portfolio Management**: Add/Remove positions with weighted average cost basis.
- **Risk Analytics**: Portfolio Beta, Sharpe Ratio, Volatility, and Max Drawdown.
- **Correlation Matrix**: Dynamic heatmap of asset correlations.
- **Local Cache**: Persistent SQLite storage for lightning-fast performance and API rate-limit protection.
