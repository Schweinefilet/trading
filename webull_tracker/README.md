# Webull CSV to Excel Auto-Sync

Automated trading journal that monitors your Downloads folder for Webull CSV exports and merges them into a professional Excel log with deduplication and statistics.

## Features
- **Auto-detection**: Instantly picks up new `TradeHistory*.csv`, `OrderHistory*.csv`, etc.
- **Smart Deduplication**: Prevents importing the same trade twice using content hashing.
- **Formatted Excel**: Generates a clean, color-coded Excel log (`webull_trading_log.xlsx`).
- **Summary Stats**: Tracks total trades, buy/sell counts, and unique symbols.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration** (Optional)
   Edit `config.py` if you want to change the watched folder or output path.
   - Default Watch Folder: `E:\Downloads E`
   - Default Output: `E:\trading\webull_tracker\webull_trading_log.xlsx`

3. **Run**
   ```bash
   python webull_watcher.py
   ```
   Keep the window open. The script will monitor for new files.

## Usage
Simply export your trade history from Webull (Mobile app or Desktop).
- The script will detect the new CSV in your Downloads folder.
- It will verify new records and append them to your Excel log.
- It will ignore duplicates if you export the same range again.

## Output
The Excel file will be created in your Documents folder with two sheets:
- **Trade Log**: Detailed list of all trades.
- **Summary**: High-level statistics.
