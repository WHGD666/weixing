from __future__ import annotations


def axis_starts(length: int, tile_length: int, overlap: float) -> list[int]:
    if length <= 0 or tile_length <= 0:
        raise ValueError("length and tile_length must be positive")
    if not 0.0 <= overlap < 0.9:
        raise ValueError("overlap must be in [0, 0.9)")
    actual_tile = min(length, tile_length)
    if length <= actual_tile:
        return [0]
    step = max(1, int(round(actual_tile * (1.0 - overlap))))
    starts = list(range(0, length - actual_tile + 1, step))
    final = length - actual_tile
    if starts[-1] != final:
        starts.append(final)
    return starts


def tile_windows(
    width: int, height: int, tile_size: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    if width <= 0 or height <= 0 or tile_size <= 0:
        raise ValueError("image dimensions and tile_size must be positive")
    tile_width = min(width, tile_size)
    tile_height = min(height, tile_size)
    return [
        (x, y, x + tile_width, y + tile_height)
        for y in axis_starts(height, tile_height, overlap)
        for x in axis_starts(width, tile_width, overlap)
    ]
