import concurrent.futures
from typing import Any, Callable, List
from tqdm import tqdm

def concurrent_task_executor(task: Callable[[Any], None], data_list: List[Any], max_workers: int = 32, description: str = None) -> None:
    """
    Execute tasks concurrently on a list of data objects using ThreadPoolExecutor.
    Args:
        task (Callable): The function to apply to each data object.
        data_list (List): The list of data objects.
        max_workers (int): The maximum number of worker threads (default is 32).
        description (str, optional): Description for the progress bar.
    Raises:
        ValueError: If data_list is empty.
    Example:
        >>> def process_data(data):
        >>>     # Process data here
        >>>     pass
        >>> data_list = [1, 2, 3, 4, 5]
        >>> concurrent_task_executor(process_data, data_list, max_workers=8, description="Processing data")
    """

    if not data_list:
        raise ValueError("Data list is empty. No tasks to execute.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(task, data) for data in data_list]

        # Create progress bar
        with tqdm(total=len(data_list), desc=description) as pbar:
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                pbar.update(1)  # Update progress bar

    # Clear the data_list after all tasks are completed
    data_list.clear()