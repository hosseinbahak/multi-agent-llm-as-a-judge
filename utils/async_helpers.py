# multi_agent_llm_judge/utils/async_helpers.py
import asyncio
from typing import List, Coroutine, Any
from tqdm.asyncio import tqdm
from loguru import logger

async def run_with_concurrency_limit(
    coroutines: List[Coroutine], 
    limit: int, 
    description: str = "Processing tasks"
) -> List[Any]:
    """
    Runs a list of coroutines with a specified concurrency limit.

    Args:
        coroutines: A list of awaitable coroutines to run.
        limit: The maximum number of coroutines to run at the same time.
        description: A description for the progress bar.

    Returns:
        A list of results from the coroutines. Exceptions are returned as-is
        for tasks that failed.
    """
    semaphore = asyncio.Semaphore(limit)
    results = []

    async def run_coro_with_semaphore(coro: Coroutine):
        async with semaphore:
            return await coro

    # Use return_exceptions=True to prevent one failure from stopping all tasks
    tasks = [run_coro_with_semaphore(coro) for coro in coroutines]
    
    results = await tqdm.gather(*tasks, return_exceptions=True, desc=description)
    
    # Log any exceptions that occurred
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Task {i} failed with an exception: {result}")
            # Optionally, you might want to re-raise or handle specific exceptions
            
    return results
