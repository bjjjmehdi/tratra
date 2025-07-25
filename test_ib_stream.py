import asyncio, os, sys
sys.path.insert(0, os.getcwd())

from src.data_ingestion.ib_stream import IBStreamer
from src.utils.config import load_config

async def main():
    cfg = load_config()
    cfg["ib"]["client_id"] = 888  # pick unused
    stream = IBStreamer(cfg["ib"])
    print("[1] Connecting...")
    await stream.connect()
    print("[2] Connected:", stream.ib.isConnected())

    if not stream.ib.isConnected():
        return

    print("[3] Subscribing to tick stream...")
    async for tick in stream.tick_stream():
        print("[4] Tick:", tick.contract.symbol, tick.last, "@", tick.time)
        break  # stop after first tick

if __name__ == "__main__":
    asyncio.run(main())