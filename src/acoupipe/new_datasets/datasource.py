import builtins
import functools
from copy import copy
from typing import (
    TYPE_CHECKING,
    Iterable,
    List,
    Optional,
    Tuple,
)

import numpy as np
from ray.data import read_datasource
from ray.data.block import Block, BlockAccessor, BlockMetadata
from ray.data.context import DataContext
from ray.data.dataset import Dataset
from ray.data.datasource import Datasource, ReadTask

if TYPE_CHECKING:

    pass

class RangeDatasource(Datasource):
    """A datasource that generates ranges of numbers from [start..stop)."""

    def __init__(
        self,
        start: int,
        stop: int,
        block_format: str = "arrow",
        tensor_shape: Tuple = (1,),
        column_name: Optional[str] = None,
        name: Optional[str] = None,
    ):
        self._start = int(start)
        self._stop = int(stop)
        self._block_format = block_format
        self._tensor_shape = tensor_shape
        self._column_name = column_name
        self._name = name

    def estimate_inmemory_data_size(self) -> Optional[int]:
        if self._block_format == "tensor":
            element_size = int(np.prod(self._tensor_shape))
        else:
            element_size = 1
        return 8 * (self._stop - self._start) * element_size

    def get_read_tasks(
        self,
        parallelism: int,
    ) -> List[ReadTask]:
        read_tasks: List[ReadTask] = []
        start = self._start
        stop = self._stop
        block_format = self._block_format
        tensor_shape = self._tensor_shape
        total_count = stop - start
        block_size = max(1, total_count // parallelism)

        ctx = DataContext.get_current()
        if total_count == 0:
            target_rows_per_block = 0
        else:
            row_size_bytes = self.estimate_inmemory_data_size() // total_count
            row_size_bytes = max(row_size_bytes, 1)
            target_rows_per_block = max(1, ctx.target_max_block_size // row_size_bytes)

        def make_block(start: int, count: int) -> Block:
            if block_format == "arrow":
                import pyarrow as pa

                return pa.Table.from_arrays(
                    [np.arange(start, start + count)],
                    names=[self._column_name or "value"],
                )
            elif block_format == "tensor":
                import pyarrow as pa

                tensor = np.ones(tensor_shape, dtype=np.int64) * np.expand_dims(
                    np.arange(start, start + count),
                    tuple(range(1, 1 + len(tensor_shape))),
                )
                return BlockAccessor.batch_to_block(
                    {self._column_name: tensor} if self._column_name else tensor
                )
            else:
                return list(builtins.range(start, start + count))

        def make_blocks(
            start: int, count: int, target_rows_per_block: int
        ) -> Iterable[Block]:
            while count > 0:
                num_rows = min(count, target_rows_per_block)
                yield make_block(start, num_rows)
                start += num_rows
                count -= num_rows

        if block_format == "tensor":
            element_size = int(np.prod(tensor_shape))
        else:
            element_size = 1

        i = start
        while i < stop:
            count = min(block_size, stop - i)
            meta = BlockMetadata(
                num_rows=count,
                size_bytes=8 * count * element_size,
                schema=copy(self._schema()),
                input_files=None,
                exec_stats=None,
            )
            read_tasks.append(
                ReadTask(
                    lambda i=i, count=count: make_blocks(
                        i, count, target_rows_per_block
                    ),
                    meta,
                )
            )
            i += block_size

        return read_tasks

    @functools.cache
    def _schema(self):
        if self._start == self._stop:
            return None

        if self._block_format == "arrow":
            import pyarrow as pa

            schema = pa.Table.from_pydict({self._column_name or "value": [0]}).schema
        elif self._block_format == "tensor":
            import pyarrow as pa

            tensor = np.ones(self._tensor_shape, dtype=np.int64) * np.expand_dims(
                np.arange(0, 10), tuple(range(1, 1 + len(self._tensor_shape)))
            )
            schema = BlockAccessor.batch_to_block(
                {self._column_name: tensor} if self._column_name else tensor
            ).schema
        elif self._block_format == "list":
            schema = int
        else:
            raise ValueError("Unsupported block type", self._block_format)
        return schema

    def get_name(self) -> str:
        return self._name


def range(
    start: int,
    stop: int,
    name: Optional[str] = None,
    *,
    parallelism: int = -1,
    concurrency: Optional[int] = None,
    override_num_blocks: Optional[int] = None,
) -> Dataset:
    """Create a :class:`~ray.data.Dataset`.

    Returns
    -------
        A :class:`~ray.data.Dataset` producing the integers from the range start to stop.

    .. seealso::

        :meth:`~ray.data.range_tensor`
                    Call this method for creating synthetic datasets of tensor data.

    """
    datasource = RangeDatasource(
        start=start, stop=stop, block_format="arrow", column_name="idx")
    if name is not None:
        datasource._name = name
    return read_datasource(
        datasource,
        parallelism=parallelism,
        concurrency=concurrency,
        override_num_blocks=override_num_blocks,
        #ray_remote_args={"num_cpus": 0.25},
    )


if __name__ == "__main__":
    ds = range(start=10, stop=12)
    print(ds.take_all())
