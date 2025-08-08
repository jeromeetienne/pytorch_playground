import random
import time

# from https://www.labri.fr/perso/nrougier/from-python-to-numpy/#code-vectorization


class ScreenHelper:

    @staticmethod
    def clear_screen():
        print("\033c", end="")  # ANSI escape code to clear the console

    def red_string(s: str) -> str:
        return f"\033[31m{s}\033[0m"

    def green_string(s: str) -> str:
        return f"\033[32m{s}\033[0m"


class GameOfLife:

    @staticmethod
    def initial_state_random(
        state_width=10,
        state_height=10,
        initial_living_cell_count=None,
        random_seed=None,
    ):
        if initial_living_cell_count is None:
            initial_living_cell_count = (state_width * state_height) // 2

        # setup an empty initial state
        state_initial = [[0 for _ in range(state_width)] for _ in range(state_height)]

        # specify the seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        # add initial live cells at random positions
        for _ in range(initial_living_cell_count):
            x = random.randint(0, state_width - 1)
            y = random.randint(0, state_height - 1)
            state_initial[y][x] = 1

        # setup a random initial state
        # state_initial = [[random.randint(0, 1) for _ in range(state_width)] for _ in range(state_height)]

        return state_initial

    @staticmethod
    def initial_state_original():
        """
        Original initial state from the Game of Life example.

        from https://www.labri.fr/perso/nrougier/from-python-to-numpy/#code-vectorization
        """
        initial_state = [[0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]]
        return initial_state

    @staticmethod
    def initial_state_all_oscillator():
        """
        Oscillator initial state (Blinker) for the Game of Life.
        """
        # define a 20x20 grid with only 0, in an expanded form
        initial_state = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        return initial_state

    @staticmethod
    def compute_neighbours(
        current_state_zero_bordered: list[list[int]],
    ) -> list[list[int]]:
        """
        Compute the number of neighbours for each cell in the Game of Life grid.

        Args:
            Z (list[list[int]]): The current state of the Game of Life. (with a border of zeros)
        Returns:
            list[list[int]]: A grid of the same size as Z, where each cell contains
                            the number of live neighbours for the corresponding cell in Z.
        """
        shape = len(current_state_zero_bordered), len(current_state_zero_bordered[0])
        neighbors_count_arr = [
            [
                0,
            ]
            * (shape[0])
            for i in range(shape[1])
        ]
        for x in range(1, shape[0] - 1):
            for y in range(1, shape[1] - 1):
                neighbors_count_arr[x][y] = (
                    current_state_zero_bordered[x - 1][y - 1]
                    + current_state_zero_bordered[x][y - 1]
                    + current_state_zero_bordered[x + 1][y - 1]
                    + current_state_zero_bordered[x - 1][y]
                    + current_state_zero_bordered[x + 1][y]
                    + current_state_zero_bordered[x - 1][y + 1]
                    + current_state_zero_bordered[x][y + 1]
                    + current_state_zero_bordered[x + 1][y + 1]
                )
        return neighbors_count_arr

    @staticmethod
    def state_iterate(current_state_zero_bordered):
        """
        Iterate the state of the Game of Life.

        Args:
            current_state_zero_bordered (list[list[int]]): The current state of the Game of Life. (with a border of zeros)
        """
        shape = len(current_state_zero_bordered), len(current_state_zero_bordered[0])
        neighbors_count_arr = GameOfLife.compute_neighbours(current_state_zero_bordered)
        for x in range(1, shape[0] - 1):
            for y in range(1, shape[1] - 1):
                if current_state_zero_bordered[x][y] == 1 and (
                    neighbors_count_arr[x][y] < 2 or neighbors_count_arr[x][y] > 3
                ):
                    current_state_zero_bordered[x][y] = 0
                elif current_state_zero_bordered[x][y] == 0 and neighbors_count_arr[x][y] == 3:
                    current_state_zero_bordered[x][y] = 1
        return current_state_zero_bordered

    @staticmethod
    def state_print_current(current_state_zero_bordered):
        # Remove the first and last row, and first and last column from each row
        current_state = [row[1:-1] for row in current_state_zero_bordered[1:-1]]
        for row in current_state:
            # Display 1 as 'X' and 0 as '.'
            print(
                "".join(
                    (
                        ScreenHelper.red_string("* ")
                        if cell == 1
                        else ScreenHelper.green_string(". ")
                    )
                    for cell in row
                )
            )

    @staticmethod
    def state_add_zero_border(Z):
        state_width = len(Z[0])
        # Add a border of zeros around the initial state
        return (
            [[0] * (state_width + 2)]
            + [[0] + row + [0] for row in Z]
            + [[0] * (state_width + 2)]
        )

    @staticmethod
    def state_run_generation(
        state_initial, state_generation_count=50, delay_inter_generation_seconds=0.1
    ):
        """
        Run the Game of Life for a given number of generations.

        Args:
            state_initial (list[list[int]]): The initial state of the Game of Life.
            state_generation_count (int): The number of generations to run.
        """

        # Add a border of zeros around the initial state to ease the neighbour counting
        state_zero_bordered = GameOfLife.state_add_zero_border(state_initial)

        # clear the console
        ScreenHelper.clear_screen()

        print(f"initial state:")
        GameOfLife.state_print_current(state_zero_bordered)
        # wait 1 second before the next iteration
        time.sleep(delay_inter_generation_seconds)

        for iteration_count in range(state_generation_count):
            # clear the console
            ScreenHelper.clear_screen()

            state_zero_bordered = GameOfLife.state_iterate(state_zero_bordered)
            print(f"After {iteration_count+1} iterations:")
            GameOfLife.state_print_current(state_zero_bordered)

            # wait 1 second before the next iteration
            time.sleep(delay_inter_generation_seconds)


#################################################################

# state_initial = GameOfLife.initial_state_random(state_width=40, state_height=40, initial_living_cell_count=400, random_seed=None)
# state_initial = GameOfLife.initial_state_original()
state_initial = GameOfLife.initial_state_all_oscillator()

GameOfLife.state_run_generation(
    state_initial=state_initial,
    # state_generation_count=4,
    # delay_inter_generation_seconds=2,
)
