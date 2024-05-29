# db_zoo

这个库包含db_graph中的一些节点以及识别模型类。

安装

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
pip install --force-reinstall dist/db_-*-py3-none-any.whl
```

如果需要做开发，建议使用编辑模式来导入项目：

```bash
# pip
pip install -e PATH_TO_THIS_FOLDER
# pdm
pdm add --dev -e PATH_TO_THIS_FOLDER
```

**请不要将包提交到公开pypi server**，后续会更新到私有pypi server。

## 使用

在使用了db_graph的项目中import本项目的节点类即可在配置文件中添加对应节点。
