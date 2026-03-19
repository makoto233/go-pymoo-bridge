# Black Box Optimization Service

基于 FastAPI + Pymoo 的黑盒优化服务。

服务本身不计算适应度，而是通过 Ask-and-Tell 机制与外部系统协作：

1. 外部系统调用 `/init` 获取候选解 `X`
2. 外部系统自行计算目标值 `F`，以及可选约束 `G`
3. 外部系统调用 `/step` 将 `X/F/G` 回传
4. Python 服务继续生成下一代，直到优化结束

## 功能特性

- 单目标与多目标统一支持
- 支持可选约束 `G`
- 基于统一 `PymooWrapper` 适配不同算法
- 当前内置算法：`ga`、`nsga2`、`nsga3`、`pso`
- 基于内存字典的任务管理
- 默认 1 小时未活跃任务自动清理
- 自动生成 OpenAPI 文档与 Swagger UI

## 本地启动

先进入项目目录，然后运行：

- 安装依赖：`pip install fastapi pydantic uvicorn pymoo numpy`
- 启动服务：`uvicorn main:app --host 0.0.0.0 --port 8000`

启动后可访问：

- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Docker 使用

项目已包含以下容器相关文件：

- [Dockerfile](Dockerfile)
- [.dockerignore](.dockerignore)

### 1. 构建镜像

```bash
docker build -t moo-blackbox-service .
```

### 2. 运行容器

```bash
docker run --rm -p 8000:8000 moo-blackbox-service
```

服务启动后访问：

- `http://127.0.0.1:8000/docs`

## 测试

当前包含两类简单测试：

- 算法流程测试：直接调用 Python 接口函数
- 前端通信测试：通过 HTTP 请求模拟前端调用 `/init` 和 `/step`

运行方式：

```bash
python -m unittest discover -s tests -v
```

## API 说明

### `POST /init`

创建一个新的优化任务，并返回初始种群。

示例请求：

```json
{
	"algorithm": "nsga3",
	"n_var": 3,
	"n_obj": 3,
	"xl": 0,
	"xu": 1,
	"pop_size": 6,
	"n_gen": 10,
	"n_ieq_constr": 0,
	"seed": 1,
	"verbose": false,
	"algorithm_params": {
		"n_partitions": 2
	}
}
```

示例响应：

```json
{
	"task_id": "f5c0db6e-7d2f-4ab0-b2d7-a77e1c6a4a5d",
	"x": [
		[0.1, 0.5, 0.2],
		[0.8, 0.3, 0.9]
	]
}
```

### `POST /step`

提交外部系统已计算好的 `X/F/G`，并获取下一代种群或最终最优解。

单目标示例请求：

```json
{
	"task_id": "f5c0db6e-7d2f-4ab0-b2d7-a77e1c6a4a5d",
	"x": [
		[0.1, 0.5, 0.2],
		[0.8, 0.3, 0.9]
	],
	"f": [1.25, 0.91]
}
```

多目标带约束示例请求：

```json
{
	"task_id": "f5c0db6e-7d2f-4ab0-b2d7-a77e1c6a4a5d",
	"x": [
		[0.1, 0.5, 0.2],
		[0.8, 0.3, 0.9]
	],
	"f": [
		[1.25, 3.7],
		[0.91, 2.8]
	],
	"g": [
		[0.0],
		[1.2]
	]
}
```

进行中示例响应：

```json
{
	"task_id": "f5c0db6e-7d2f-4ab0-b2d7-a77e1c6a4a5d",
	"done": false,
	"generation": 2,
	"next_x": [
		[0.3, 0.7, 0.4],
		[0.6, 0.2, 0.8]
	],
	"best_x": [
		[0.8, 0.3, 0.9]
	],
	"best_f": [
		[0.91]
	],
	"best_g": null
}
```

结束示例响应：

```json
{
	"task_id": "f5c0db6e-7d2f-4ab0-b2d7-a77e1c6a4a5d",
	"done": true,
	"generation": 11,
	"next_x": null,
	"best_x": [
		[0.8, 0.3, 0.9]
	],
	"best_f": [
		[0.91]
	],
	"best_g": null
}
```

## 字段约定

- `n_obj = 1` 时，`f` 可以传一维数组，也可以传二维数组
- `n_obj > 1` 时，`f` 会按 `(-1, n_obj)` 自动 reshape
- 配置了 `n_ieq_constr > 0` 时，`g` 为必填
- `xl` / `xu` 支持标量或长度为 `n_var` 的数组
- 当任务完成后，对应 `task_id` 会自动移除

## 典型集成流程

1. Go 服务调用 `/init`
2. 获取 `task_id` 和 `x`
3. Go 服务计算每个解的 `f` / `g`
4. 调用 `/step`
5. 若 `done=false`，继续使用返回的 `next_x`
6. 若 `done=true`，读取最终 `best_x`、`best_f`、`best_g`

## 如何把代码打包给前端

推荐交付 3 份内容：

1. **可运行后端包**
	- 直接交付 Docker 镜像，前端或联调同学只需要执行容器
	- 或交付源码压缩包：包含 [Dockerfile](Dockerfile)、[README.md](README.md)、[pyproject.toml](pyproject.toml)

2. **接口契约**
	- 提供 `http://<host>:8000/openapi.json`
	- 前端可基于 OpenAPI 自动生成类型定义和调用代码

3. **联调示例**
	- 给前端 `/init` 和 `/step` 的 JSON 示例
	- 明确 `task_id`、`x`、`f`、`g` 的字段结构

### 推荐交付方式

如果是给前端联调，最省事的是直接交付 Docker 镜像：

```bash
docker build -t moo-blackbox-service .
docker save -o moo-blackbox-service.tar moo-blackbox-service
```

然后把 `moo-blackbox-service.tar` 发给前端，前端本地执行：

```bash
docker load -i moo-blackbox-service.tar
docker run --rm -p 8000:8000 moo-blackbox-service
```

这样前端只需要对接：

- `POST /init`
- `POST /step`
- `GET /docs`
- `GET /openapi.json`

### 如果交付源码

建议把以下文件一起打包：

- [main.py](main.py)
- [algorithms/__init__.py](algorithms/__init__.py)
- [algorithms/base.py](algorithms/base.py)
- [algorithms/pymoo_wrapper.py](algorithms/pymoo_wrapper.py)
- [state_manager.py](state_manager.py)
- [tests/test_main.py](tests/test_main.py)
- [pyproject.toml](pyproject.toml)
- [Dockerfile](Dockerfile)
- [.dockerignore](.dockerignore)
- [README.md](README.md)

## 容器启动说明

当前 [Dockerfile](Dockerfile) 默认使用：

- 基础镜像：`python:3.14-slim`
- 启动命令：`uvicorn main:app --host 0.0.0.0 --port 8000`

如需调整端口，可在运行容器时修改映射关系，例如：

```bash
docker run --rm -p 18000:8000 moo-blackbox-service
```
