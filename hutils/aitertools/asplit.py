from typing import AsyncIterable, Iterable
import asyncio

class SubAsyncIterable:
    def __init__(self):
        self.waiting = asyncio.Event()
        self.has_next = asyncio.Event()
        self.element = None
        self.exception = None
    def _new_element(self, e):
        self.element = e
        self.has_next.set()
    def _set_exception(self, e):
        self.exception = e
        self.has_next.set()
    def __aiter__(self):
        return self
    async def __anext__(self):
        self.waiting.set()
        await self.has_next.wait()
        self.has_next.clear()
        if self.exception:
            raise self.exception
        e = self.element
        self.element = None
        return e

async def asplit(aiter: AsyncIterable[Iterable]):
    es = await aiter.__anext__()
    subs = [SubAsyncIterable() for _ in range(len(es))]

    async def fetch_all():
        async def new_e(e, sub):
            await sub.waiting.wait()
            sub.waiting.clear()
            sub._new_element(e)
        def new_es(es):
            return asyncio.wait([asyncio.create_task(new_e(e, sub)) for e, sub in zip(es, subs)])

        await new_es(es)
        while True:
            try:
                n_es = await aiter.__anext__()
                await new_es(n_es)
            except Exception as e:
                for sub in subs:
                    sub._set_exception(e)
                break
    asyncio.create_task(fetch_all())
    return subs
