# Multi-Stage Build Separation Plan

## Intent

将当前的单镜像 TensorRT 构建流程拆分为 Builder/Runtime 双镜像架构:
- **Builder**: 使用 NVIDIA 官方 `trtexec` CLI 工具 (ONNX → .engine), 无需自写 Python builder 脚本
- **Runtime**: 基于 `nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04`, 仅包含推理运行时依赖, 加载 `.engine` 文件执行推理

目标: 最终交付的 Runtime 镜像体积最小、攻击面最小, 不含编译工具链。

## Feasibility

**可行** - `trtexec` 是 NVIDIA 随 TensorRT 分发的生产级 C++ CLI 工具:
- 包含在 `nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04` 的 TensorRT 安装中
- 路径: `/usr/src/tensorrt/bin/trtexec` (deb/tar 安装) 或 `/opt/tensorrt/bin/trtexec` (NGC 容器)
- 完全覆盖 `inference_smoke_test.py` 中 `build_engine()` 的所有功能
- 额外支持: 动态 shape 配置、INT8 量化、benchmark、per-layer profiling

## Context

### 为什么用 trtexec 替代自写 Python builder

| 维度 | 自写 `build_engine.py` | `trtexec` |
|------|------------------------|-----------|
| 代码量 | ~80 行 Python | 0 行 (二进制工具) |
| 依赖 | tensorrt, numpy, yaml, Python runtime | 无 (纯 C++, 使用 libnvinfer) |
| 镜像体积 | 需要 Python + pip 包 | 无额外开销 |
| 维护 | 自己维护 bug/兼容性 | NVIDIA 官方维护 |
| 功能覆盖 | 基本 FP16 builder | 全部 TensorRT knob: FP16/INT8/动态 shape/calibration/tactic profiling |
| 验证能力 | 无 | 内置 benchmark + latency 统计 |

**结论**: 自写 builder 脚本是在重复造轮子。trtexec 是 NVIDIA 为此场景专门设计的工具。

### 现状分析

| 文件 | 现状 | 问题 |
|------|------|------|
| `dockerfiles/direct_trt.Dockerfile` | 使用 `devel` 镜像, TensorRT 安装被注释 | 功能不完整, 仅安装了 PyTorch |
| `Makefile` | 单一 `build` target | 无法区分构建阶段 |
| `dockerfiles/inference_smoke_test.py` | `build_engine()` + `run_inference()` 耦合 | 职责不分, 顶层 pycuda import |
| `.dockerignore` | 排除 `Dockerfile*`, `*.sh`, `Makefile` | 多 Dockerfile 构建时需调整 |

### trtexec 关键用法 (来自 NVIDIA 官方文档)

```bash
# ONNX → .engine (Builder 阶段)
trtexec --onnx=model.onnx \
        --saveEngine=model.engine \
        --fp16

# .engine → benchmark (验证阶段)
trtexec --loadEngine=model.engine \
        --shapes=input:1x14
```

核心 flags:
| Flag | 说明 |
|------|------|
| `--onnx=FILE` | 输入 ONNX 模型 |
| `--saveEngine=FILE` | 输出 .engine 文件 |
| `--loadEngine=FILE` | 加载已有 .engine (benchmark/验证) |
| `--fp16` | 启用 FP16 精度 |
| `--int8` | 启用 INT8 量化 |
| `--shapes=NAME:DIMxDIM` | 指定输入 tensor shape |
| `--minShapes`, `--optShapes`, `--maxShapes` | 动态 shape profile |
| `--workspace=N` | workspace 大小 (MB) |
| `--verbose` | 详细日志 |

## Task

### Parent Task
实现 trtexec-based Builder/Runtime 双镜像架构, 通过 Makefile 统一管理构建流程。

### Child Tasks

#### Child 1: 创建 run_inference.py - Runtime 推理脚本

**Delivers**: `dockerfiles/run_inference.py`

**策略: 直接替换 `inference_smoke_test.py`, 不保留原文件**

Builder 阶段不再需要 Python 脚本 (trtexec 替代)。只保留一个 Runtime 推理脚本:

**`run_inference.py`** — Runtime 阶段:
- 从 `inference_smoke_test.py` 的 `prepare_buffers()` + `run_inference()` 提取
- 依赖: `tensorrt`, `pycuda`, `numpy`, `yaml`
- 使用 `trt.Runtime().deserialize_cuda_engine()` 从 `.engine` 文件加载 (不从 ONNX 构建)
- 删除 `build_engine()` 方法 — 这是 trtexec 的职责
- 保留 dummy input 生成和 smoke test 流程
- CLI 入口: `python3 run_inference.py --engine-dir /opt/engine --model-name vtol_hover`

**`inference_smoke_test.py`** — 删除

**依赖**: 无
**完成标志**: `run_inference.py` 可独立运行, 无 build_engine 相关代码, 无 pycuda import 问题

#### Child 2: 创建 Builder Dockerfile

**Delivers**: `dockerfiles/builder.Dockerfile`

```dockerfile
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04

# 安装 TensorRT (trtexec 随附)
# 安装: tensorrt, tensorrt-libs (trtexec 二进制在 /usr/src/tensorrt/bin/trtexec)

# 不需要 Python, 不需要 pycuda, 不需要 PyTorch
# 不需要 build-essential, cmake (trtexec 是预编译二进制)

# COPY models/ 到 /opt/models/

# ENTRYPOINT: trtexec
#   --onnx=/opt/models/$MODEL_NAME/model.onnx
#   --saveEngine=/opt/engine/$MODEL_NAME.engine
#   --fp16
```

- 基础镜像: `nvidia/cuda:12.5.1-cudnn-devel-ubuntu22.04`
- 仅安装 TensorRT deb 包 (含 trtexec)
- COPY `models/` 到 `/opt/models/`
- 不安装: Python, pip, pycuda, PyTorch, build-essential, cmake
- ENTRYPOINT: `trtexec --onnx=... --saveEngine=... --fp16`
- 产物: `/opt/engine/*.engine` 通过 volume 挂载导出

**依赖**: 无 (trtexec 是已有工具)
**完成标志**: `docker build -f builder.Dockerfile .` 成功, trtexec 可用

#### Child 3: 创建 Runtime Dockerfile

**Delivers**: `dockerfiles/runtime.Dockerfile`

- 基础镜像: `nvidia/cuda:12.5.1-cudnn-runtime-ubuntu22.04`
- 安装: Python 3.10, pip, `tensorrt`, `pycuda`, `numpy`, `pyyaml`
- COPY `run_inference.py` 到 `/opt/app/`
- COPY `.engine` 文件到 `/opt/engine/` (从 builder 产物)
- 不含: nvcc, trtexec, build-essential, cmake, PyTorch, ONNX
- CMD: `python3 /opt/app/run_inference.py`

**依赖**: Child 1
**完成标志**: `docker build -f runtime.Dockerfile .` 成功, 镜像体积显著小于 builder

#### Child 4: 扩展 Makefile

**Delivers**: 更新后的 `Makefile`

新增变量:
```makefile
BUILDER_IMAGE ?= trt-builder
RUNTIME_IMAGE ?= trt-runtime
BUILDER_DOCKERFILE := $(DOCKERFILE_DIR)/builder.Dockerfile
RUNTIME_DOCKERFILE := $(DOCKERFILE_DIR)/runtime.Dockerfile
ENGINE_OUTPUT_DIR ?= ./engine-output
TRTEXEC_PRECISION ?= fp16
```

新增 targets:
- `build-builder` — 构建 builder 镜像
- `build-engine` — 运行 builder 容器, trtexec ONNX→.engine, 产物输出到 `ENGINE_OUTPUT_DIR`
- `build-runtime` — 构建 runtime 镜像, 将 `.engine` 文件 bake 进镜像
- `build-all` — 依次执行 `build-builder` → `build-engine` → `build-runtime`
- `test-runtime` — 使用 runtime 镜像运行推理 smoke test
- `clean-all` — 清理所有镜像和产物

`build-engine` target 的核心逻辑:
```makefile
build-engine: build-builder prepare-models
	docker run --gpus all --rm \
	  -v $(CURDIR)/$(ENGINE_OUTPUT_DIR):/opt/engine \
	  -v $(CURDIR)/$(LOCAL_MODEL_DIR):/opt/models \
	  $(BUILDER_IMAGE):$(IMAGE_TAG)
```

保留 targets (兼容):
- `build` → `build-all`
- `test` → `test-runtime`
- `clean` → `clean-all`

**依赖**: Child 2, Child 3
**完成标志**: `make build-all` 完整走通, `make test-runtime` 推理成功

#### Child 5: 更新 .dockerignore

**Delivers**: 更新后的 `.dockerignore`

- 从排除列表中移除 `Dockerfile*`
- 保留排除 `*.sh`, `Makefile`
- 添加 `engine-output/` 到排除列表

**依赖**: 无
**完成标志**: `docker build` 上下文大小合理

#### Child 6: README.md 添加 trtexec 使用指南

**Delivers**: `README.md` 中新增 `trtexec` 使用指南章节

内容:
- trtexec 简介 (NVIDIA 官方 TensorRT CLI 工具)
- 基本 ONNX→Engine 转换命令
- FP16 / INT8 精度选择
- 动态 shape 配置
- Benchmark 用法
- 与本项目 Makefile 的集成说明

**依赖**: 无
**完成标志**: README.md 包含可操作的 trtexec 示例命令

## Deliverables

| # | Deliverable | Owner |
|---|-------------|-------|
| 1 | `dockerfiles/run_inference.py` — Engine 推理脚本 | Child 1 |
| 2 | `dockerfiles/builder.Dockerfile` — trtexec 编译镜像 | Child 2 |
| 3 | `dockerfiles/runtime.Dockerfile` — 推理运行时镜像 | Child 3 |
| 4 | Makefile 更新 — 多阶段构建 targets | Child 4 |
| 5 | `.dockerignore` 更新 | Child 5 |
| 6 | `README.md` — trtexec 使用指南 | Child 6 |

## Acceptance

- [ ] `make build-all` 成功构建 builder 和 runtime 两个镜像
- [ ] `make test-runtime` 在 runtime 镜像上完成推理 smoke test
- [ ] Runtime 镜像体积 < Builder 镜像体积的 60%
- [ ] Runtime 镜像不含 nvcc, trtexec, build-essential, cmake, PyTorch
- [ ] Builder 镜像不含 Python, pycuda, PyTorch
- [ ] `inference_smoke_test.py` 已删除

## Constraints

### 依赖矩阵 (更新)

| 依赖 | Builder (trtexec) | Runtime (run_inference.py) |
|------|:---:|:---:|
| `trtexec` (C++ binary) | ✅ | ❌ |
| `tensorrt` (Python) | ❌ | ✅ |
| `pycuda` | ❌ | ✅ |
| `numpy` | ❌ | ✅ |
| `pyyaml` | ❌ | ✅ |
| `onnx` (Python) | ❌ | ❌ |
| `Python` | ❌ | ✅ |

Builder 阶段完全不需要 Python runtime — trtexec 是纯 C++ 二进制。

### 其他约束

- `inference_smoke_test.py` 直接删除
- Builder 和 Runtime 的 CUDA/cuDNN 版本必须匹配 (12.5.1)
- `.engine` 文件是 GPU 架构特定的, 不可跨平台移植
- Builder 仍需 GPU 设备 (trtexec 需要 GPU 做图优化), 通过 `docker run --gpus` 传入

## Rules

- 使用 `DOCKER_BUILDKIT=1` 构建
- 所有 image name/tag 可通过 Makefile 变量覆盖
- Python 版本统一 3.10
- 基础镜像 tag 通过 `TAG` 变量控制
- trtexec 精度默认 FP16, 可通过 `TRTEXEC_PRECISION` 变量覆盖
