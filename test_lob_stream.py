import asyncio, os, sys
sys.path.insert(0, os.getcwd())

from src.data_ingestion.lob_stream import LobStream
from src.utils.config import load_config
from ib_insync import Stock

async def main():
    from src.utils.config import load_config
    cfg = load_config()
    ib = cfg["ib"]
    from ib_insync import IB
    ibc = IB()
    await ibc.connectAsync(host=ib["host"], port=ib["port"], clientId=889)
    contract = Stock("INTC", "SMART", "USD")
    lob = LobStream(ibc)
    print("[1] Subscribing to LOB...")
    async for l in lob.stream(contract):
        print("[2] LOB:", l.bid[:2], l.ask[:2], "latency_us", l.latency_us)
        break

if __name__ == "__main__":
    asyncio.run(main())