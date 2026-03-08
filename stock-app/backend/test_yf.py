import yfinance as yf
ticker = yf.Ticker("AMZN")
hist = ticker.history(period="5d", interval="1h")
print(hist.head(2))
print("Index type:", type(hist.index))
print("Index name:", hist.index.name)
