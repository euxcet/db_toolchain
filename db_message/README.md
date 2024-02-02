# db_message

这是用于在不同位置共享消息文件的工具，实现方式是在本地维护一些在线仓库（如Github Repository），并在项目中用配置文件指定消息文件的来源。本工具可以被视为一个简单的包管理工具，因此也可被用于其他用途。

本工具的使用场景是当两个应用项目需要维护一批共享的消息文件，但又希望消息文件能够在另一个消息仓库中被统一管理，同时能在应用项目中完成同步和版本控制。之所以不直接在应用项目里直接做版本控制，是有以下原因：

1. 如果两个应用项目在不同的仓库中或在同一个仓库中引用不同的消息文件，则需要手动同步共享的文件，操作繁琐且易于出错。

2. 如果两个应用项目在同一个仓库中且引用同一个消息文件，会出现引用路径过长、被某一方改动导致其他方无法使用的问题。

3. 如果同一个消息文件的不同历史版本被各种应用项目引用会造成管理上的困难。

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
pip install --force-reinstall dist/db_message-*-py3-none-any.whl
```

**请不要将包提交到公开pypi server**，后续会更新到私有pypi server。

## 使用

安装之后会获得一个名为dbm的终端命令，如果是在pdm虚拟环境中安装需要使用pdm run dbm来调用。通过以下指令来检查是否安装成功：

```bash
dbm --help
# if installed within pdm's virtual environment.
pdm run dbm --help
```

dbm维护两种类型的文件，一类是从在线或其他同步方式获得的仓库（目前仅支持Git仓库），另一类是从使用dbm工具同步应用项目中的模块，其中需同步的文件都是仓库中某个时间点的某个文件的拷贝。

### 仓库操作

仓库操作都在dbm repo中：

```bash
dbm repo --help
```

其中包含以下几类操作：

- 克隆在线仓库

```bash
dbm repo clone URL
```

- 列举本地仓库

```bash
dbm repo list
```

- 更新本地仓库

```bash
dbm repo update NAME
```

- 查看本地仓库信息

```bash
dbm repo info NAME
```

**请不要在本地仓库中做修改或提交，这可能会导致dbm工具无法与在线仓库保持同步！**

### 模块操作

- 创建模块

```bash
dbm create [--path <path>]
```

根据指引完成创建后会在模块文件夹中生成dbm_module.json配置文件，模块中除了dbm_module.json的其他文件都会默认被.gitignore忽略，如果有需要请自行修改忽略规则。

- 添加文件

首先进入模块文件夹或在参数中给出模块路径。

```bash
dbm add [--path <path>]
```

根据指引完成添加后会将文件拷贝到模块文件夹并加入配置文件中，之后根据Compiler选项做对应的编译，如果不需要额外的编译可在Compiler选项中选择TextCompiler。在配置时可使用tab键补全或查看候选项。

- 删除文件

```bash
dbm remove NAME [--path <path>]
```

删除对指定文件的维护。

- 更新文件

```bash
dbm update [--name <name>]
           [--path <path>]
```

更新整个模块或某些文件。