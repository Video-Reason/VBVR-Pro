"""Self-contained random maze layout + video renderer for synthetic RL data.

Produces:
    - a random perfect maze on a fixed logical grid
      (``1 = wall``, ``0 = passage``);
    - a BFS shortest path from start to goal;
    - a sequence of RGB frames showing either a ball travelling that path or
      a path line being drawn over a configurable number of frames;
    - all metadata needed by a custom reward (grid, start, goal, path,
      per-frame path position, palette, pixel geometry, generation settings).

No torch / CUDA dependencies — this module is pure Python + numpy + PIL so it
can be imported by the precompute pipeline, the reward implementation, and
offline visualisation tools.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------


class MazePalette(BaseModel):
    """Named RGB palette used to render one maze sample.

    Colors are 0-255 uint8 triplets.  Names are only used to build the text
    prompt and to record which palette was sampled.
    """

    wall_rgb: tuple[int, int, int]
    passage_rgb: tuple[int, int, int]
    ball_rgb: tuple[int, int, int]
    goal_rgb: tuple[int, int, int]
    wall_name: str
    passage_name: str
    ball_name: str
    goal_name: str


DEFAULT_PALETTES: list[MazePalette] = [
    MazePalette(
        wall_rgb=(20, 20, 20),
        passage_rgb=(240, 240, 240),
        ball_rgb=(220, 40, 40),
        goal_rgb=(40, 180, 80),
        wall_name="black",
        passage_name="white",
        ball_name="red",
        goal_name="green",
    ),
    MazePalette(
        wall_rgb=(30, 50, 90),
        passage_rgb=(235, 230, 210),
        ball_rgb=(240, 170, 30),
        goal_rgb=(180, 60, 60),
        wall_name="navy",
        passage_name="cream",
        ball_name="orange",
        goal_name="crimson",
    ),
    MazePalette(
        wall_rgb=(50, 50, 50),
        passage_rgb=(210, 220, 235),
        ball_rgb=(60, 120, 230),
        goal_rgb=(240, 200, 60),
        wall_name="slate",
        passage_name="ice",
        ball_name="blue",
        goal_name="gold",
    ),
    MazePalette(
        wall_rgb=(60, 35, 20),
        passage_rgb=(245, 235, 215),
        ball_rgb=(200, 60, 160),
        goal_rgb=(70, 170, 110),
        wall_name="brown",
        passage_name="parchment",
        ball_name="magenta",
        goal_name="emerald",
    ),
]


RenderMode = Literal["moving_ball", "growing_path_line"]

RENDER_MODE_MOVING_BALL = "moving_ball"
RENDER_MODE_GROWING_PATH_LINE = "growing_path_line"
RENDER_MODE_ALIASES: dict[str, RenderMode] = {
    "moving_ball": RENDER_MODE_MOVING_BALL,
    "ball": RENDER_MODE_MOVING_BALL,
    "growing_path_line": RENDER_MODE_GROWING_PATH_LINE,
    "path_line": RENDER_MODE_GROWING_PATH_LINE,
    "line": RENDER_MODE_GROWING_PATH_LINE,
}


def normalize_render_mode(mode: str) -> RenderMode:
    key = mode.strip().lower().replace("-", "_")
    try:
        return RENDER_MODE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(set(RENDER_MODE_ALIASES)))
        raise ValueError(f"Unknown maze render mode '{mode}'. Valid values: {valid}") from exc


# ---------------------------------------------------------------------------
# Difficulty schedule
# ---------------------------------------------------------------------------


class MazeDifficulty(BaseModel):
    """Random-maze difficulty recipe.

    ``path_ratio`` is relative to the start-goal Manhattan distance on the
    rendered logical grid.  The generator rejects mazes whose true BFS path
    falls outside this range.
    """

    id: int
    name: str
    path_ratio_min: float
    path_ratio_max: float
    branch_count_range: tuple[int, int]
    branch_len_range: tuple[int, int]
    prompt_adjective: str


DEFAULT_DIFFICULTIES: tuple[MazeDifficulty, ...] = (
    MazeDifficulty(
        id=0,
        name="easy",
        path_ratio_min=1.00,
        path_ratio_max=2.25,
        branch_count_range=(0, 0),
        branch_len_range=(0, 0),
        prompt_adjective="simple",
    ),
    MazeDifficulty(
        id=1,
        name="mid",
        path_ratio_min=2.25,
        path_ratio_max=3.60,
        branch_count_range=(0, 0),
        branch_len_range=(0, 0),
        prompt_adjective="moderately winding",
    ),
    MazeDifficulty(
        id=2,
        name="hard",
        path_ratio_min=3.60,
        path_ratio_max=5.00,
        branch_count_range=(0, 0),
        branch_len_range=(0, 0),
        prompt_adjective="complex",
    ),
    MazeDifficulty(
        id=3,
        name="xhard",
        path_ratio_min=5.00,
        path_ratio_max=6.80,
        branch_count_range=(0, 0),
        branch_len_range=(0, 0),
        prompt_adjective="very difficult",
    ),
)

DIFFICULTY_BY_NAME: dict[str, MazeDifficulty] = {d.name: d for d in DEFAULT_DIFFICULTIES}
DIFFICULTY_BY_NAME["medium"] = DIFFICULTY_BY_NAME["mid"]


# ---------------------------------------------------------------------------
# Maze layout + path
# ---------------------------------------------------------------------------


def generate_maze_grid(cell_h: int, cell_w: int, rng: np.random.Generator) -> np.ndarray:
    """Recursive-backtracker perfect maze.

    Cells live at odd grid coordinates, walls at even coordinates.  The
    returned array has shape ``(2*cell_h + 1, 2*cell_w + 1)`` and dtype uint8
    (``1 = wall``, ``0 = passage``).
    """
    H, W = 2 * cell_h + 1, 2 * cell_w + 1
    grid = np.ones((H, W), dtype=np.uint8)

    visited = np.zeros((cell_h, cell_w), dtype=bool)
    stack: list[tuple[int, int]] = [(0, 0)]
    visited[0, 0] = True
    grid[1, 1] = 0

    while stack:
        ci, cj = stack[-1]
        # Unvisited neighbours in random order.
        neighbours: list[tuple[int, int, int, int]] = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < cell_h and 0 <= nj < cell_w and not visited[ni, nj]:
                neighbours.append((ni, nj, di, dj))
        if not neighbours:
            stack.pop()
            continue
        pick = int(rng.integers(0, len(neighbours)))
        ni, nj, di, dj = neighbours[pick]
        # Carve the wall between (ci, cj) and (ni, nj) as well as the cell.
        grid[2 * ci + 1 + di, 2 * cj + 1 + dj] = 0
        grid[2 * ni + 1, 2 * nj + 1] = 0
        visited[ni, nj] = True
        stack.append((ni, nj))

    return grid


_DIRS_4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _sample_endpoint_pair(
    grid_h: int,
    grid_w: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Sample start near the top-left and goal near the bottom-right."""
    row_jitter = max(2, grid_h // 8)
    col_jitter = max(2, grid_w // 8)
    start = (
        int(rng.integers(1, min(grid_h - 1, row_jitter + 1))),
        int(rng.integers(1, min(grid_w - 1, col_jitter + 1))),
    )
    goal = (
        int(rng.integers(max(1, grid_h - row_jitter - 1), grid_h - 1)),
        int(rng.integers(max(1, grid_w - col_jitter - 1), grid_w - 1)),
    )
    return start, goal


def _weighted_pick(weights: list[float], rng: np.random.Generator) -> int:
    total = float(sum(weights))
    if total <= 0:
        return int(rng.integers(0, len(weights)))
    r = float(rng.random() * total)
    acc = 0.0
    for idx, weight in enumerate(weights):
        acc += float(weight)
        if acc >= r:
            return idx
    return len(weights) - 1


def _random_self_avoiding_path(
    grid_h: int,
    grid_w: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    min_len: int,
    max_len: int,
    rng: np.random.Generator,
    max_steps: int,
) -> tuple[list[tuple[int, int]] | None, int]:
    """Find a random simple path with length in ``[min_len, max_len]``.

    The search is a weighted DFS with backtracking.  Before the lower bound is
    met it prefers detours; after that it prefers moving toward the goal.
    """
    path: list[tuple[int, int]] = [start]
    visited: set[tuple[int, int]] = {start}
    tried_stack: list[set[tuple[int, int]]] = [set()]

    steps = 0
    while path and steps < max_steps:
        steps += 1
        cur = path[-1]
        path_len = len(path) - 1

        if cur == goal:
            if min_len <= path_len <= max_len:
                return path, steps
            # Goal reached too early.  Backtrack; leaving and revisiting the
            # goal would violate the simple-path constraint.
            candidates: list[tuple[int, int]] = []
            weights: list[float] = []
        else:
            cur_dist = _manhattan(cur, goal)
            candidates = []
            weights = []
            for di, dj in _DIRS_4:
                nb = (cur[0] + di, cur[1] + dj)
                if nb in tried_stack[-1]:
                    continue
                ni, nj = nb
                if not (0 <= ni < grid_h and 0 <= nj < grid_w):
                    continue
                if nb in visited:
                    continue
                visited_neighbours = 0
                for ndi, ndj in _DIRS_4:
                    if (ni + ndi, nj + ndj) in visited:
                        visited_neighbours += 1
                if visited_neighbours != 1:
                    continue

                new_len = path_len + 1
                new_dist = _manhattan(nb, goal)
                if new_len + new_dist > max_len:
                    continue
                if nb == goal and new_len < min_len:
                    continue

                delta = new_dist - cur_dist
                if path_len < min_len:
                    # Positive delta moves away from the goal and creates
                    # harder paths.  Near the max bound, pull back toward goal.
                    weight = float(np.exp(0.9 * delta))
                    if new_len + new_dist > max_len * 0.85:
                        weight *= float(np.exp(-1.5 * delta))
                else:
                    weight = float(np.exp(-1.2 * delta))
                candidates.append(nb)
                weights.append(max(weight, 1e-3))

        if candidates:
            pick = _weighted_pick(weights, rng)
            nb = candidates[pick]
            tried_stack[-1].add(nb)
            path.append(nb)
            visited.add(nb)
            tried_stack.append(set())
            continue

        old = path.pop()
        visited.remove(old)
        tried_stack.pop()

    return None, steps


def _open_neighbour_count(grid: np.ndarray, cell: tuple[int, int]) -> int:
    grid_h, grid_w = grid.shape
    ci, cj = cell
    count = 0
    for di, dj in _DIRS_4:
        ni, nj = ci + di, cj + dj
        if 0 <= ni < grid_h and 0 <= nj < grid_w and grid[ni, nj] == 0:
            count += 1
    return count


def _scale_count_for_area(count_range: tuple[int, int], grid_h: int, grid_w: int) -> tuple[int, int]:
    scale = (grid_h * grid_w) / float(48 * 48)
    lo = max(1, int(round(count_range[0] * scale)))
    hi = max(lo, int(round(count_range[1] * scale)))
    return lo, hi


def _carve_dead_end_branches(
    grid: np.ndarray,
    rng: np.random.Generator,
    branch_count: int,
    branch_len_range: tuple[int, int],
) -> int:
    """Carve random dead ends while avoiding loops/shortcuts."""
    open_cells = [tuple(map(int, p)) for p in np.argwhere(grid == 0)]
    carved = 0
    for _ in range(branch_count):
        if not open_cells:
            break
        root = open_cells[int(rng.integers(0, len(open_cells)))]
        cur = root
        branch_len = int(rng.integers(branch_len_range[0], branch_len_range[1] + 1))
        for _step in range(branch_len):
            candidates: list[tuple[int, int]] = []
            for di, dj in _DIRS_4:
                nb = (cur[0] + di, cur[1] + dj)
                ni, nj = nb
                if not (0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]):
                    continue
                if grid[ni, nj] == 0:
                    continue
                if _open_neighbour_count(grid, nb) != 1:
                    continue
                candidates.append(nb)
            if not candidates:
                break
            cur = candidates[int(rng.integers(0, len(candidates)))]
            grid[cur] = 0
            open_cells.append(cur)
            carved += 1
    return carved


def _resolve_difficulty(name: str) -> MazeDifficulty:
    try:
        return DIFFICULTY_BY_NAME[name]
    except KeyError as exc:
        valid = ", ".join(sorted(DIFFICULTY_BY_NAME))
        raise ValueError(f"Unknown maze difficulty '{name}'. Valid values: {valid}") from exc


def generate_random_maze_grid(
    grid_h: int,
    grid_w: int,
    difficulty: MazeDifficulty,
    rng: np.random.Generator,
    *,
    max_attempts: int = 64,
    max_search_steps: int = 250_000,
    sample_seed: int | None = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], list[tuple[int, int]], dict[str, Any]]:
    """Generate a random corridor maze and return full generation metadata."""
    last_steps = 0
    for attempt in range(max_attempts):
        start, goal = _sample_endpoint_pair(grid_h, grid_w, rng)
        manhattan = _manhattan(start, goal)
        min_len = max(manhattan, int(np.ceil(manhattan * difficulty.path_ratio_min)))
        max_len = max(min_len, int(np.ceil(manhattan * difficulty.path_ratio_max)))
        max_len = min(max_len, grid_h * grid_w - 1)

        path, search_steps = _random_self_avoiding_path(
            grid_h,
            grid_w,
            start,
            goal,
            min_len,
            max_len,
            rng,
            max_search_steps,
        )
        last_steps = search_steps
        if path is None:
            continue

        grid = np.ones((grid_h, grid_w), dtype=np.uint8)
        for cell in path:
            grid[cell] = 0

        branch_count_lo, branch_count_hi = _scale_count_for_area(difficulty.branch_count_range, grid_h, grid_w)
        branch_count = int(rng.integers(branch_count_lo, branch_count_hi + 1))
        branch_cells = _carve_dead_end_branches(grid, rng, branch_count, difficulty.branch_len_range)

        # Branches are loop-free by construction, but run BFS anyway so the
        # stored path is always the real shortest path under the final grid.
        shortest_path = bfs_shortest_path(grid, start, goal)
        path_len = len(shortest_path) - 1
        path_ratio = path_len / max(1, manhattan)
        generation = {
            "method": "random_self_avoiding_corridor_with_dead_end_branches",
            "difficulty": difficulty.name,
            "difficulty_id": difficulty.id,
            "sample_seed": sample_seed,
            "attempt": attempt,
            "search_steps": search_steps,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "start_goal_strategy": "random_jittered_top_left_to_bottom_right",
            "manhattan_distance": manhattan,
            "target_path_len_min": min_len,
            "target_path_len_max": max_len,
            "branch_count": branch_count,
            "branch_cells": branch_cells,
            "branch_len_min": difficulty.branch_len_range[0],
            "branch_len_max": difficulty.branch_len_range[1],
            "path_len": path_len,
            "path_ratio": path_ratio,
            "wall_fraction": float(grid.mean()),
            "open_fraction": float(1.0 - grid.mean()),
        }
        return grid, start, goal, shortest_path, generation

    raise RuntimeError(
        "Could not generate random maze after "
        f"{max_attempts} attempts for difficulty={difficulty.name}; last_search_steps={last_steps}"
    )


def generate_perfect_maze_grid(
    cell_h: int,
    cell_w: int,
    difficulty: MazeDifficulty,
    rng: np.random.Generator,
    *,
    max_attempts: int = 256,
    sample_seed: int | None = None,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], list[tuple[int, int]], dict[str, Any]]:
    """Generate a clear, one-cell-wide perfect maze at shape ``(2h, 2w)``.

    ``generate_maze_grid`` returns the textbook ``(2h+1, 2w+1)`` layout with
    an outer wall.  Cropping the top/left border gives an exactly divisible
    ``(2h, 2w)`` logical grid, so ``cell_h=cell_w=16`` and ``cell_px=12``
    render to 384x384 while preserving the normal maze topology.
    """
    start = (0, 0)
    goal = (2 * cell_h - 2, 2 * cell_w - 2)
    manhattan = _manhattan(start, goal)
    last_path_len = -1
    last_ratio = -1.0

    for attempt in range(max_attempts):
        grid = generate_maze_grid(cell_h, cell_w, rng)[1:, 1:]
        path = bfs_shortest_path(grid, start, goal)
        path_len = len(path) - 1
        path_ratio = path_len / max(1, manhattan)
        last_path_len = path_len
        last_ratio = path_ratio

        if difficulty.path_ratio_min <= path_ratio < difficulty.path_ratio_max:
            turns = 0
            for a, b, c in zip(path, path[1:], path[2:], strict=False):
                d1 = (b[0] - a[0], b[1] - a[1])
                d2 = (c[0] - b[0], c[1] - b[1])
                if d1 != d2:
                    turns += 1
            generation = {
                "method": "perfect_dfs_maze_cropped_to_even_grid",
                "difficulty": difficulty.name,
                "difficulty_id": difficulty.id,
                "sample_seed": sample_seed,
                "attempt": attempt,
                "cell_h": cell_h,
                "cell_w": cell_w,
                "grid_h": int(grid.shape[0]),
                "grid_w": int(grid.shape[1]),
                "start_goal_strategy": "top_left_room_to_bottom_right_room",
                "manhattan_distance": manhattan,
                "target_path_ratio_min": difficulty.path_ratio_min,
                "target_path_ratio_max": difficulty.path_ratio_max,
                "branch_count": 0,
                "branch_cells": 0,
                "branch_len_min": 0,
                "branch_len_max": 0,
                "path_len": path_len,
                "path_ratio": path_ratio,
                "turn_count": turns,
                "wall_fraction": float(grid.mean()),
                "open_fraction": float(1.0 - grid.mean()),
            }
            return grid, start, goal, path, generation

    raise RuntimeError(
        "Could not generate perfect maze after "
        f"{max_attempts} attempts for difficulty={difficulty.name}; "
        f"last_path_len={last_path_len} last_path_ratio={last_ratio:.3f}"
    )


def bfs_shortest_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    """4-neighbour BFS on passage cells. ``start``/``goal`` are grid coords."""
    H, W = grid.shape
    if grid[start] == 1 or grid[goal] == 1:
        raise ValueError("start or goal falls on a wall cell")

    parents: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        ci, cj = cur
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] == 0 and (ni, nj) not in parents:
                parents[(ni, nj)] = cur
                q.append((ni, nj))

    if goal not in parents:
        raise RuntimeError("BFS could not reach goal — generated maze is disconnected")

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell_center_xy(i: int | float, j: int | float, cell_px: int) -> tuple[float, float]:
    """Pixel (x, y) of the centre of grid cell (i, j)."""
    return (j * cell_px + cell_px / 2.0, i * cell_px + cell_px / 2.0)


def _goal_marker_half_extent(cell_px: int) -> float:
    return cell_px / 3.0


def _moving_ball_radius(cell_px: int) -> float:
    return cell_px / 3.0


def _path_line_width(cell_px: int) -> int:
    return max(3, int(round(cell_px * 0.38)))


def _path_start_cap_radius(cell_px: int) -> int:
    return max(2, int(round(cell_px * 0.22)))


def _completion_frame(num_frames: int, completion_fraction: float) -> int:
    """Frame index where a progressive path should be complete."""
    if not 0.0 < completion_fraction <= 1.0:
        raise ValueError(f"completion_fraction must be in (0, 1], got {completion_fraction}")
    if num_frames <= 1:
        return 0
    return max(1, min(num_frames - 1, int(round((num_frames - 1) * completion_fraction))))


def interpolate_ball_positions(
    path: list[tuple[int, int]],
    num_frames: int,
    cell_px: int,
    *,
    completion_frame: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate the ball along ``path`` over ``num_frames`` frames.

    Returns ``(frame_positions_cell, frame_positions_pix)``, each shape
    ``(num_frames, 2)`` as float32.  ``cell`` coordinates are (i, j) row/col,
    ``pix`` coordinates are (x, y) pixel.
    """
    assert len(path) >= 1
    L = len(path)
    if completion_frame is None:
        completion_frame = num_frames - 1
    completion_frame = max(0, min(num_frames - 1, int(completion_frame)))
    frame_positions_cell = np.zeros((num_frames, 2), dtype=np.float32)
    frame_positions_pix = np.zeros((num_frames, 2), dtype=np.float32)
    for f in range(num_frames):
        t = float(L - 1) if completion_frame <= 0 else min(float(L - 1), f * (L - 1) / completion_frame)
        k = int(np.floor(t))
        alpha = float(t - k)
        if k >= L - 1:
            ci, cj = path[-1]
            frame_positions_cell[f] = (ci, cj)
        else:
            ci = path[k][0] * (1 - alpha) + path[k + 1][0] * alpha
            cj = path[k][1] * (1 - alpha) + path[k + 1][1] * alpha
            frame_positions_cell[f] = (ci, cj)
        x, y = _cell_center_xy(frame_positions_cell[f, 0], frame_positions_cell[f, 1], cell_px)
        frame_positions_pix[f] = (x, y)
    return frame_positions_cell, frame_positions_pix


def render_frame(
    grid: np.ndarray,
    ball_xy_pix: tuple[float, float],
    goal_ij: tuple[int, int],
    palette: MazePalette,
    cell_px: int,
) -> np.ndarray:
    """Render one RGB frame. Returns uint8 array of shape (H, W, 3)."""
    H_cells, W_cells = grid.shape
    img_h = H_cells * cell_px
    img_w = W_cells * cell_px

    img = Image.new("RGB", (img_w, img_h), palette.passage_rgb)
    draw = ImageDraw.Draw(img)

    # Walls — run-length compress along rows to reduce draw calls.
    for i in range(H_cells):
        j = 0
        while j < W_cells:
            if grid[i, j] == 1:
                j0 = j
                while j < W_cells and grid[i, j] == 1:
                    j += 1
                draw.rectangle(
                    [j0 * cell_px, i * cell_px, j * cell_px - 1, (i + 1) * cell_px - 1],
                    fill=palette.wall_rgb,
                )
            else:
                j += 1

    # Goal marker — filled square a bit smaller than a cell.
    gi, gj = goal_ij
    gcx, gcy = _cell_center_xy(gi, gj, cell_px)
    gr = cell_px / 3.0
    draw.rectangle(
        [gcx - gr, gcy - gr, gcx + gr, gcy + gr],
        fill=palette.goal_rgb,
    )

    # Ball — filled circle.
    bx, by = ball_xy_pix
    br = cell_px / 3.0
    draw.ellipse(
        [bx - br, by - br, bx + br, by + br],
        fill=palette.ball_rgb,
    )

    return np.asarray(img, dtype=np.uint8)


def render_maze_base(
    grid: np.ndarray,
    goal_ij: tuple[int, int],
    palette: MazePalette,
    cell_px: int,
    *,
    goal_marker_half_extent_px: float | None = None,
) -> Image.Image:
    """Render maze walls/background/goal without the animated foreground."""
    H_cells, W_cells = grid.shape
    img_h = H_cells * cell_px
    img_w = W_cells * cell_px

    img = Image.new("RGB", (img_w, img_h), palette.passage_rgb)
    draw = ImageDraw.Draw(img)

    for i in range(H_cells):
        j = 0
        while j < W_cells:
            if grid[i, j] == 1:
                j0 = j
                while j < W_cells and grid[i, j] == 1:
                    j += 1
                draw.rectangle(
                    [j0 * cell_px, i * cell_px, j * cell_px - 1, (i + 1) * cell_px - 1],
                    fill=palette.wall_rgb,
                )
            else:
                j += 1

    gi, gj = goal_ij
    gcx, gcy = _cell_center_xy(gi, gj, cell_px)
    gr = _goal_marker_half_extent(cell_px) if goal_marker_half_extent_px is None else goal_marker_half_extent_px
    draw.rectangle(
        [gcx - gr, gcy - gr, gcx + gr, gcy + gr],
        fill=palette.goal_rgb,
    )
    return img


def render_video(
    grid: np.ndarray,
    frame_positions_pix: np.ndarray,
    goal_ij: tuple[int, int],
    palette: MazePalette,
    cell_px: int,
) -> np.ndarray:
    """Render every frame. Returns uint8 array of shape (T, H, W, 3)."""
    frames = [render_frame(grid, (float(x), float(y)), goal_ij, palette, cell_px) for x, y in frame_positions_pix]
    return np.stack(frames, axis=0)


def render_path_line_video(
    grid: np.ndarray,
    frame_positions_pix: np.ndarray,
    path: list[tuple[int, int]],
    goal_ij: tuple[int, int],
    palette: MazePalette,
    cell_px: int,
    *,
    line_rgb: tuple[int, int, int] | None = None,
    line_width_px: int | None = None,
    start_cap_radius_px: int | None = None,
    goal_marker_half_extent_px: float | None = None,
    line_completion_frame: int | None = None,
) -> np.ndarray:
    """Render a video where the solution path is drawn progressively."""
    line_color = palette.ball_rgb if line_rgb is None else line_rgb
    line_width = _path_line_width(cell_px) if line_width_px is None else line_width_px
    start_radius = _path_start_cap_radius(cell_px) if start_cap_radius_px is None else start_cap_radius_px
    base = render_maze_base(
        grid,
        goal_ij,
        palette,
        cell_px,
        goal_marker_half_extent_px=goal_marker_half_extent_px,
    )
    path_centers = [_cell_center_xy(i, j, cell_px) for i, j in path]

    frames: list[np.ndarray] = []
    n_frames = int(frame_positions_pix.shape[0])
    if line_completion_frame is None:
        line_completion_frame = n_frames - 1
    line_completion_frame = max(0, min(n_frames - 1, int(line_completion_frame)))
    for frame_idx, cur_xy in enumerate(frame_positions_pix):
        img = base.copy()
        draw = ImageDraw.Draw(img)
        if line_completion_frame <= 0:
            t = float(len(path_centers) - 1)
        else:
            t = min(float(len(path_centers) - 1), frame_idx * (len(path_centers) - 1) / line_completion_frame)
        k = int(np.floor(t))
        cur = (float(cur_xy[0]), float(cur_xy[1]))
        points = path_centers[: max(1, k + 1)]
        if not points or points[-1] != cur:
            points = points + [cur]
        if len(points) >= 2:
            draw.line(points, fill=line_color, width=line_width, joint="curve")

        sx, sy = path_centers[0]
        draw.ellipse(
            [sx - start_radius, sy - start_radius, sx + start_radius, sy + start_radius],
            fill=line_color,
        )
        frames.append(np.asarray(img, dtype=np.uint8))

    return np.stack(frames, axis=0)


def render_video_from_metadata(maze: dict[str, Any]) -> np.ndarray:
    """Reconstruct the RGB video frames from the JSON ``maze`` metadata blob."""
    palette_blob = maze["palette"]
    palette = MazePalette(
        wall_rgb=tuple(palette_blob["wall_rgb"]),
        passage_rgb=tuple(palette_blob["passage_rgb"]),
        ball_rgb=tuple(palette_blob["ball_rgb"]),
        goal_rgb=tuple(palette_blob["goal_rgb"]),
        wall_name=palette_blob["wall_name"],
        passage_name=palette_blob["passage_name"],
        ball_name=palette_blob["ball_name"],
        goal_name=palette_blob["goal_name"],
    )
    grid = np.asarray(maze["grid"], dtype=np.uint8)
    frame_positions_pix = np.asarray(maze["frame_positions_pix"], dtype=np.float32)
    goal = tuple(int(x) for x in maze["goal"])
    path = [tuple(int(x) for x in p) for p in maze["path"]]
    cell_px = int(maze["cell_px"])
    render_mode = normalize_render_mode(maze.get("render_mode", RENDER_MODE_MOVING_BALL))
    render_metadata = maze.get("render_metadata") or {}

    if render_mode == RENDER_MODE_MOVING_BALL:
        return render_video(grid, frame_positions_pix, goal, palette, cell_px)

    if render_mode == RENDER_MODE_GROWING_PATH_LINE:
        return render_path_line_video(
            grid,
            frame_positions_pix,
            path,
            goal,
            palette,
            cell_px,
            line_rgb=tuple(render_metadata.get("line_rgb", palette.ball_rgb)),
            line_width_px=int(render_metadata.get("line_width_px", _path_line_width(cell_px))),
            start_cap_radius_px=int(render_metadata.get("start_cap_radius_px", _path_start_cap_radius(cell_px))),
            goal_marker_half_extent_px=float(
                render_metadata.get("goal_marker_half_extent_px", _goal_marker_half_extent(cell_px))
            ),
            line_completion_frame=(
                int(render_metadata["line_completion_frame"]) if "line_completion_frame" in render_metadata else None
            ),
        )

    raise AssertionError(f"Unhandled render mode: {render_mode}")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


PROMPT_TEMPLATES: tuple[str, ...] = (
    "A {ball} ball navigates through a {difficulty} {wall} maze, following the valid path to the goal.",
    "A small {ball} circle moves through a {difficulty} maze with {wall} walls on a "
    "{passage} background, heading toward the {goal} goal marker.",
    "Top-down view of a {difficulty} {passage} maze with {wall} walls; a {ball} ball "
    "traverses the shortest path to the {goal} exit.",
    "A {ball} ball rolls step-by-step through a {difficulty} grid maze, navigating "
    "{wall} corridors to reach the {goal} target at the far corner.",
)

LINE_PROMPT_TEMPLATES: tuple[str, ...] = (
    "A {ball} path line is drawn through a {difficulty} {wall} maze to the {goal} goal.",
    "Top-down view of a {difficulty} {passage} maze with {wall} walls; a {ball} line "
    "traces the path to the {goal} exit.",
    "A {ball} solution line grows step-by-step through a {difficulty} grid maze toward the {goal} target.",
)


def build_prompt(palette: MazePalette, difficulty: MazeDifficulty, rng: np.random.Generator) -> str:
    tpl = PROMPT_TEMPLATES[int(rng.integers(0, len(PROMPT_TEMPLATES)))]
    return tpl.format(
        ball=palette.ball_name,
        wall=palette.wall_name,
        passage=palette.passage_name,
        goal=palette.goal_name,
        difficulty=difficulty.prompt_adjective,
    )


def build_line_prompt(palette: MazePalette, difficulty: MazeDifficulty, rng: np.random.Generator) -> str:
    tpl = LINE_PROMPT_TEMPLATES[int(rng.integers(0, len(LINE_PROMPT_TEMPLATES)))]
    return tpl.format(
        ball=palette.ball_name,
        wall=palette.wall_name,
        passage=palette.passage_name,
        goal=palette.goal_name,
        difficulty=difficulty.prompt_adjective,
    )


def build_line_waypoint_from_sample(
    sample: MazeSample,
    *,
    completion_fraction: float = 0.5,
) -> tuple[np.ndarray, MazeSample]:
    """Render a path-line variant from an existing maze sample.

    The variant uses the same maze, palette, and prompt as the source sample,
    but its path line reaches the goal by ``completion_fraction`` of the video.
    """
    line_completion_frame = _completion_frame(sample.num_frames, completion_fraction)
    frame_positions_cell, frame_positions_pix = interpolate_ball_positions(
        sample.path,
        sample.num_frames,
        sample.cell_px,
        completion_frame=line_completion_frame,
    )
    line_width_px = _path_line_width(sample.cell_px)
    start_cap_radius_px = int(round(_moving_ball_radius(sample.cell_px)))
    goal_marker_half_extent_px = _goal_marker_half_extent(sample.cell_px)
    grid = np.asarray(sample.grid, dtype=np.uint8)
    video = render_path_line_video(
        grid,
        frame_positions_pix,
        sample.path,
        sample.goal,
        sample.palette,
        sample.cell_px,
        line_width_px=line_width_px,
        start_cap_radius_px=start_cap_radius_px,
        goal_marker_half_extent_px=goal_marker_half_extent_px,
        line_completion_frame=line_completion_frame,
    )
    waypoint = sample.model_copy(
        deep=True,
        update={
            "frame_positions_cell": [tuple(row) for row in frame_positions_cell.tolist()],
            "frame_positions_pix": [tuple(row) for row in frame_positions_pix.tolist()],
            "render_mode": RENDER_MODE_GROWING_PATH_LINE,
            "render_metadata": {
                "renderer_version": 1,
                "mode": RENDER_MODE_GROWING_PATH_LINE,
                "line_rgb": list(sample.palette.ball_rgb),
                "line_width_px": line_width_px,
                "start_cap_shape": "circle",
                "start_cap_radius_px": start_cap_radius_px,
                "goal_rgb": list(sample.palette.goal_rgb),
                "goal_marker_shape": "square",
                "goal_marker_half_extent_px": goal_marker_half_extent_px,
                "line_completion_fraction": completion_fraction,
                "line_completion_frame": line_completion_frame,
            },
        },
    )
    return video, waypoint


# ---------------------------------------------------------------------------
# End-to-end sample builder
# ---------------------------------------------------------------------------


class MazeSpec(BaseModel):
    """Configuration for one maze sample generator.

    Resolution is derived as ``(2*cell_h)*cell_px`` by ``(2*cell_w)*cell_px``.
    """

    cell_h: int = Field(ge=2)
    cell_w: int = Field(ge=2)
    cell_px: int = Field(ge=4)
    num_frames: int = Field(ge=2)
    palettes: list[MazePalette] = Field(default_factory=lambda: list(DEFAULT_PALETTES))
    difficulty_names: tuple[str, ...] = ("easy", "mid", "hard", "xhard")
    render_mode: str = RENDER_MODE_MOVING_BALL
    max_generation_attempts: int = Field(default=64, ge=1)
    max_search_steps: int = Field(default=250_000, ge=1)

    @property
    def image_hw(self) -> tuple[int, int]:
        return (2 * self.cell_h * self.cell_px, 2 * self.cell_w * self.cell_px)


class MazeSample(BaseModel):
    """Fully-specified maze sample — everything the reward might need."""

    # Layout
    grid: list[list[int]]  # shape (Hg, Wg), 1=wall 0=passage
    grid_h: int
    grid_w: int
    cell_h: int
    cell_w: int
    cell_px: int
    image_h: int
    image_w: int

    # Task
    start: tuple[int, int]  # (i, j) in grid coords
    goal: tuple[int, int]  # (i, j) in grid coords
    path: list[tuple[int, int]]  # BFS shortest path (start → goal inclusive)
    path_len: int
    manhattan_distance: int
    path_ratio: float

    # Animation
    num_frames: int
    frame_positions_cell: list[tuple[float, float]]  # per-frame (i, j) in cell coords
    frame_positions_pix: list[tuple[float, float]]  # per-frame (x, y) in pixels
    render_mode: str = RENDER_MODE_MOVING_BALL
    render_metadata: dict[str, Any] = Field(default_factory=dict)

    # Difficulty + generation metadata
    difficulty: str
    difficulty_id: int
    generation: dict[str, Any]

    # Palette + text
    palette: MazePalette
    prompt: str


def build_maze_sample(
    spec: MazeSpec,
    rng: np.random.Generator,
    *,
    sample_seed: int | None = None,
) -> tuple[np.ndarray, MazeSample]:
    """Generate maze + render video + return full metadata.

    The start is sampled near the top-left and the goal near the bottom-right.
    Difficulty controls the random corridor path length and branch density.

    Returns ``(video_uint8, MazeSample)`` where ``video_uint8`` has shape
    ``(num_frames, H, W, 3)``.
    """
    if not spec.difficulty_names:
        raise ValueError("MazeSpec.difficulty_names must contain at least one difficulty")
    difficulty_name = spec.difficulty_names[int(rng.integers(0, len(spec.difficulty_names)))]
    difficulty = _resolve_difficulty(difficulty_name)

    grid, start, goal, path, generation = generate_perfect_maze_grid(
        spec.cell_h,
        spec.cell_w,
        difficulty,
        rng,
        max_attempts=spec.max_generation_attempts,
        sample_seed=sample_seed,
    )

    fp_cell, fp_pix = interpolate_ball_positions(path, spec.num_frames, spec.cell_px)

    palette = spec.palettes[int(rng.integers(0, len(spec.palettes)))]

    render_mode = normalize_render_mode(spec.render_mode)
    if render_mode == RENDER_MODE_MOVING_BALL:
        video = render_video(grid, fp_pix, goal, palette, spec.cell_px)
        prompt = build_prompt(palette, difficulty, rng)
        render_metadata = {
            "renderer_version": 1,
            "mode": render_mode,
            "ball_rgb": list(palette.ball_rgb),
            "ball_radius_px": _moving_ball_radius(spec.cell_px),
            "goal_rgb": list(palette.goal_rgb),
            "goal_marker_shape": "square",
            "goal_marker_half_extent_px": _goal_marker_half_extent(spec.cell_px),
        }
    elif render_mode == RENDER_MODE_GROWING_PATH_LINE:
        line_width_px = _path_line_width(spec.cell_px)
        start_cap_radius_px = _path_start_cap_radius(spec.cell_px)
        goal_marker_half_extent_px = _goal_marker_half_extent(spec.cell_px)
        video = render_path_line_video(
            grid,
            fp_pix,
            path,
            goal,
            palette,
            spec.cell_px,
            line_width_px=line_width_px,
            start_cap_radius_px=start_cap_radius_px,
            goal_marker_half_extent_px=goal_marker_half_extent_px,
        )
        prompt = build_line_prompt(palette, difficulty, rng)
        render_metadata = {
            "renderer_version": 1,
            "mode": render_mode,
            "line_rgb": list(palette.ball_rgb),
            "line_width_px": line_width_px,
            "start_cap_shape": "circle",
            "start_cap_radius_px": start_cap_radius_px,
            "goal_rgb": list(palette.goal_rgb),
            "goal_marker_shape": "square",
            "goal_marker_half_extent_px": goal_marker_half_extent_px,
        }
    else:
        raise AssertionError(f"Unhandled render mode: {render_mode}")

    img_h, img_w = video.shape[1], video.shape[2]
    sample = MazeSample(
        grid=grid.astype(int).tolist(),
        grid_h=int(grid.shape[0]),
        grid_w=int(grid.shape[1]),
        cell_h=spec.cell_h,
        cell_w=spec.cell_w,
        cell_px=spec.cell_px,
        image_h=img_h,
        image_w=img_w,
        start=start,
        goal=goal,
        path=[tuple(p) for p in path],
        path_len=len(path),
        manhattan_distance=int(generation["manhattan_distance"]),
        path_ratio=float(generation["path_ratio"]),
        num_frames=spec.num_frames,
        frame_positions_cell=[tuple(row) for row in fp_cell.tolist()],
        frame_positions_pix=[tuple(row) for row in fp_pix.tolist()],
        render_mode=render_mode,
        render_metadata=render_metadata,
        difficulty=difficulty.name,
        difficulty_id=difficulty.id,
        generation=generation,
        palette=palette,
        prompt=prompt,
    )
    return video, sample
