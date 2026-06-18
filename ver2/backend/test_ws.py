import asyncio
import websockets

async def test():
    try:
        async with websockets.connect('ws://127.0.0.1:8000/ws/video') as ws:
            print('Connected!')
            await ws.close()
    except Exception as e:
        print(f"Failed: {e}")

asyncio.run(test())
