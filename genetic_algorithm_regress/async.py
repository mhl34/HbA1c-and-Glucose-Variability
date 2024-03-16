import asyncio
import time

async def factorial(name, number):
    f = 1
    for i in range(2, number + 1):
        print(f"Task {name}: Compute factorial({number}), currently i={i}...")
        f *= i
    await asyncio.sleep(1)
    print(f"Task {name}: factorial({number}) = {f}")
    return f

async def main():
    # Schedule three calls *concurrently*:
    print(f"started at {time.strftime('%X')}")
    L = await asyncio.gather(
        factorial("A", 2),
        factorial("B", 3),
        factorial("C", 4),
    )
    print(f"end at {time.strftime('%X')}")
    print(L)

asyncio.run(main())