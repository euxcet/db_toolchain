import sys
import numpy as np
from vispy import app, scene
from vispy.color import Color
from collections import deque
from typing import List


class SingleLinePlotter:
    def __init__(
        self,
        grid: scene.Grid,
        row: int,
        col: int,
        size: int,
        data_buffer: deque | None = None,
        data: np.ndarray | None = None,
        x_range: tuple | None = None,
        y_range: tuple | None = None,
        row_span=1,
        col_span=1,
    ) -> None:
        self.grid = grid
        self.viewbox = self.grid.add_view(
            row=row, col=col, camera="panzoom", row_span=row_span, col_span=col_span
        )
        self.viewbox.border_color = (0.5, 0.5, 0.5, 1)
        self.size = size
        self.x_range = x_range
        self.y_range = y_range

        # white line
        zero_color = np.ones((size, 4), dtype=np.float32)

        self.data = data if data is not None else np.zeros((size), dtype=np.float32)
        self.data_buffer = data_buffer

        # add x and y axis
        self.x_axis = scene.AxisWidget(orientation="bottom")
        self.x_axis.stretch = (1, 0.2)
        self.grid.add_widget(self.x_axis, row=row + 1, col=col, col_span=col_span)
        self.x_axis.link_view(self.viewbox)
        self.y_axis = scene.AxisWidget(orientation="left")
        self.y_axis.stretch = (0.2, 1)
        self.grid.add_widget(self.y_axis, row=row, col=col - 1, row_span=row_span)
        self.y_axis.link_view(self.viewbox)

        self.line = scene.visuals.Line(
            pos=self.data_to_draw, color=zero_color, parent=self.viewbox.scene
        )
        self.viewbox.add(self.line)

    @property
    def data_to_draw(self) -> np.ndarray:
        return np.array([np.linspace(0, self.size - 1, self.size), self.data]).T

    def set_range(
        self, x_range: tuple | None = None, y_range: tuple | None = None
    ) -> None:
        if x_range is None or y_range is None:
            self.viewbox.camera.set_range(x=self.x_range, y=self.y_range)
        else:
            self.viewbox.camera.set_range(x=x_range, y=y_range)

    def auto_range(self) -> None:
        self.viewbox.camera.set_range()

    def set_data(self, data: np.ndarray) -> None:
        assert data.shape == (self.size,)
        self.data = data
        self.line.set_data(pos=self.data_to_draw)

    def set_color(self, color: np.ndarray) -> None:
        self.line.set_data(color=color)

    def update(self) -> None:
        # print(self.data_buffer)
        if self.data_buffer is None:
            # raise ValueError('data_buffer is not set.')
            print("data_buffer is not set.")
        try:
            new_data = self.data_buffer.popleft()
        except IndexError:
            # print('data_buffer is empty.')
            return
        self.data = np.roll(self.data, -1, axis=0)
        self.data[-1] = new_data
        self.set_data(self.data)


class SingleBarPlotter:
    def __init__(
        self,
        grid: scene.Grid,
        row: int,
        col: int,
        size: int,
        data_buffer: deque | None = None,
        y_range: tuple | None = None,
        row_span=1,
        col_span=1,
    ) -> None:
        self.grid = grid
        self.viewbox = self.grid.add_view(
            row=row, col=col, camera="panzoom", row_span=row_span, col_span=col_span
        )
        self.viewbox.border_color = (0.5, 0.5, 0.5, 1)
        self.size = size
        self.x_range = (0, self.size + 1)
        self.y_range = y_range
        self.viewbox.camera.set_range(x=self.x_range, y=self.y_range)

        self.bar_color = Color("blue")
        self.axis_color = Color("white")

        self.data = np.zeros((size), dtype=np.float32)
        self.data_buffer = data_buffer

        self.x_axis = scene.AxisWidget(orientation="bottom")
        self.x_axis.stretch = (1, 0.2)
        self.grid.add_widget(self.x_axis, row=row + 1, col=col, col_span=col_span)
        self.x_axis.link_view(self.viewbox)
        self.y_axis = scene.AxisWidget(orientation="left")
        self.y_axis.stretch = (0.2, 1)
        self.grid.add_widget(self.y_axis, row=row, col=col - 1, row_span=row_span)
        self.y_axis.link_view(self.viewbox)

        self.bars = [
            scene.Rectangle(
                center=(i + 1, val / 2),
                height=val,
                width=0.8,
                color=self.bar_color,
                parent=self.viewbox.scene,
            )
            for i, val in enumerate(self.data)
        ]

    def set_range(
        self, x_range: tuple | None = None, y_range: tuple | None = None
    ) -> None:
        if x_range is None or y_range is None:
            self.viewbox.camera.set_range(x=self.x_range, y=self.y_range)
        else:
            self.viewbox.camera.set_range(x=x_range, y=y_range)

    def auto_range(self) -> None:
        self.viewbox.camera.set_range()

    def set_data(self, data: np.ndarray) -> None:
        assert data.shape == (self.size,)
        self.data = data
        for i, bar in enumerate(self.bars):
            bar.height = data[i]
            bar.center = (i + 1, data[i] / 2)

    def update(self) -> None:
        if self.data_buffer is None:
            raise ValueError("data_buffer is not set.")
        try:
            new_data = self.data_buffer.popleft()
        except IndexError:
            # print('data_buffer is empty.')
            return
        self.data = new_data
        self.set_data(self.data)


class MultiPlotter:
    def __init__(self) -> None:
        self.timer = app.Timer()
        self.timer.connect(self.update)
        self.canvas = scene.SceneCanvas(keys="interactive", show=True)
        self.grid = self.canvas.central_widget.add_grid(spacing=10)
        self.single_plotters: List[SingleLinePlotter] = []

    def add_single_line_plotter(
        self,
        row: int,
        col: int,
        size: int,
        data_buffer: deque | None = None,
        data: np.ndarray | None = None,
        x_range: tuple | None = None,
        y_range: tuple | None = None,
        row_span: int = 1,
        col_span: int = 1,
    ) -> None:
        new_plotter = SingleLinePlotter(
            self.grid,
            row * 2,
            col * 2 + 1,
            size,
            data_buffer=data_buffer,
            data=data,
            x_range=x_range,
            y_range=y_range,
            row_span=row_span,
            col_span=col_span,
        )
        self.single_plotters.append(new_plotter)

    def add_single_bar_plotter(
        self,
        row: int,
        col: int,
        size: int,
        data_buffer: deque | None = None,
        y_range: tuple | None = None,
        row_span: int = 1,
        col_span: int = 1,
    ) -> None:
        new_plotter = SingleBarPlotter(
            self.grid,
            row * 2,
            col * 2 + 1,
            size,
            data_buffer=data_buffer,
            y_range=y_range,
            row_span=row_span,
            col_span=col_span,
        )
        self.single_plotters.append(new_plotter)

    def update(self, ev) -> None:
        for single_plotter in self.single_plotters:
            single_plotter.update()
            single_plotter.set_range()

    def update_range(self, x_range: tuple = (0, 1), y_range: tuple = (0, 1)) -> None:
        for single_plotter in self.single_plotters:
            single_plotter.set_range(x_range, y_range)

    def start(self, interval: float | None = None) -> None:
        self.timer.start(interval)


if __name__ == "__main__" and sys.flags.interactive == 0:
    multiplotter = MultiPlotter()
    for i in range(2):
        for j in range(3):
            new_data_buffer = deque(maxlen=200)
            for k in range(200):
                new_data_buffer.append(k)
            multiplotter.add_single_line_plotter(
                i,
                j,
                200,
                data_buffer=new_data_buffer,
                x_range=(0, 200),
                y_range=(-2, 2),
            )
    data_buffer = deque(maxlen=10)
    for i in range(10):
        data_buffer.append(np.array([0.5] * 10))
    multiplotter.add_single_bar_plotter(
        2, 0, 10, y_range=(0, 1), data_buffer=data_buffer, col_span=5
    )
    multiplotter.start(1 / 120)
    app.run()