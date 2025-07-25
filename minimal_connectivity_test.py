from src.data_ingestion.ib_stream import IBStreamer
import asyncio, src.utils.config as cfg
async def test():
    s = IBStreamer(cfg.load_config()['ib'])
    await s.connect()
    print('IB connected') if s.ib.isConnected() else print('still waiting')
asyncio.run(test())