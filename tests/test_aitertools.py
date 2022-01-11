import asyncio
from hutils.aitertools import asplit

async def _dummy_iter():
    while True:
        yield 1, 2
        await asyncio.sleep(1)

async def print_iter(it):
    async for i in it:
        print(i)
        await asyncio.sleep(3 - i)

async def main():
    one, two = await asplit(_dummy_iter())
    await asyncio.wait([
        asyncio.create_task(print_iter(one)),
        asyncio.create_task(print_iter(two)),
    ])

if __name__ == '__main__':
    asyncio.run(main())
