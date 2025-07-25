from src.data_ingestion.candle_builder import CandleBuilder
from src.utils.config import load_config
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless
cfg = load_config()
builder = CandleBuilder(lookback=60)

# Fake 60 bars
df = pd.read_csv("data/replay/INTC_1m.csv", parse_dates=["timestamp"]).tail(60)
png = builder.render_png(df)
with open("test_chart.png", "wb") as f:
    f.write(png)
print("PNG written → test_chart.png")