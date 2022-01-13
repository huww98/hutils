from typing import Sequence, TypeVar, AsyncIterable, AsyncIterator, Iterable, Union
import asyncio

T = TypeVar('T')

class SubAsyncIterable(AsyncIterable[T]):
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
    async def __anext__(self) -> T:
        self.waiting.set()
        await self.has_next.wait()
        self.has_next.clear()
        if self.exception:
            raise self.exception
        e = self.element
        self.element = None
        return e

async def to_async(iterator: Iterable[T]) -> AsyncIterator[T]:
    for i in iterator:
        yield i

def ensure_async(iterator: Union[AsyncIterable[T], Iterable[T]]):
    if isinstance(iterator, Iterable):
        return to_async(iterator)
    return iterator.__aiter__()

async def asplit(aiter: Union[AsyncIterable[Sequence], Iterable[Sequence]]):
    aiter = ensure_async(aiter)
    es = await aiter.__anext__()
    subs = [SubAsyncIterable() for _ in range(len(es))]

    async def fetch_all():
        async def new_e(e, sub: SubAsyncIterable):
            await sub.waiting.wait()
            sub.waiting.clear()
            sub._new_element(e)
        def new_es(es: Sequence):
            return asyncio.gather(*(new_e(e, sub) for e, sub in zip(es, subs)))

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
