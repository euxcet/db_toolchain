# db_graph

这个工具实现了一个执行图。目的是为了连接传感器、算法、可视化等功能。对于一个常见的人机交互任务，数据会从各类传感器中不断地产生，由识别算法得到结果，再用于控制其他模块，人和其他模块会以类似中断的形式给予算法和传感器反馈控制。这种处理方式在任务简单时是线性的，但当传感器、算法等组件的增加，整个系统会变得复杂，因此需要用执行图来描述这个流程。

## 框架

### 执行图

执行图由点和边构成，点用于生产和处理数据，交互设备、识别算法、滤波、数据录制等都可以被视为图中的点。为了便于复用，点应当完成一项尽可能抽象的任务，并给出配置参数来实现可扩展性。边用于数据的传输，方式可以是内存、文件、网络等。和点类似，边中存放的数据应当有一定程度的抽象和统一，例如由指环和手套获取到的IMU数据在边中应该是相同格式的，后续的点就无需关心数据的来源。

图在执行时都是数据驱动的，每个点都在等待数据，当从边得到数据后会进行相应的计算并将结果放到输出边中。

### 配置文件

执行图的配置文件是一个json文件，定义了点的类型参数和边的连接关系。

## 安装

### PDM

项目采用[PDM](https://github.com/pdm-project/pdm)构建，这是个类似于poetry的包管理器。

PDM 需要 Python 3.8 或更高版本。

**Linux/Mac 安装命令**

```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

```bash
brew install pdm
```

```bash
pip install --user pdm
```

**Windows 安装命令**

```powershell
(Invoke-WebRequest -Uri https://pdm-project.org/install-pdm.py -UseBasicParsing).Content | python -
```

### 本库

在虚拟环境中安装本库：

```bash
pdm build
pdm install
```

在构建时需要在虚拟环境中安装包，可使用国内镜像：

```bash
pdm config pypi.url https://pypi.tuna.tsinghua.edu.cn/simple/
```

在dist文件夹中会生成whl文件，可在其他环境中使用pip安装：

```bash
pip install --force-reinstall dist/db_graph-*-py3-none-any.whl
```

**请不要将包提交到公开pypi server**，后续会更新到私有pypi server。

### 使用

本库还在不断更新中，尚未得到稳定版本，可参考实例来进行类似的开发。

请注意以下几点：

1. 实现的继承了Node的子类只需要被import引用过就会被注册，即可在配置文件中使用这个类。配置文件中用type字段来描述点是哪个类。

2. Node子类需要显式地声明输入边和输出边，例如：

```python
class GestureAggregator(Node):

  INPUT_EDGE_EVENT = 'event'
  OUTPUT_EDGE_RESULT = 'result'
```

输入边会得到相应的回调函数来处理数据，并可通过output函数来将数据放入输出边中：

```python
def handle_input_edge_event(self, event: str, timestamp: float) -> None:
  for gesture in self.gestures:
    if gesture.update(event):
      self.log_info(gesture.name)
      self.output(self.OUTPUT_EDGE_RESULT, gesture.name)
```

## 实例

### 手势识别

位于demo/demo_gesture中。

### 鼠标操控

位于demo/demo_mouse_ring中。

## TODO

- [ ] 实现不同类型的边，对于网络通信的边可构造虚拟节点。

- [ ] 用Graph类来整合NodeManager和EdgeManager。

- [ ] 将算法Node提取到demo文件夹中，并实现更多的通用Node类。

- [ ] 可视化Graph的结构，生成图片。

- [ ] 可视化传感器数据，用Node的形式。