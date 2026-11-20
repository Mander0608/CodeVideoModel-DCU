# CodeVideoModel-DCU

Code Video Model 在海光 DCU / SCNet 上的部署与实验源码：包含 MiniMax-H3 Ref2VA 注意力适配、分块 CPU offload、Three.js 无头渲染修复，以及教室和 Cabazon 场景的重访实验脚本。

基于 [Code Video Model 官方仓库](https://github.com/code-video-model/CodeVideoModel)，源码基线为 `404e03d67c2977b441622ce94411470bb84984a8`。这是部署与研究副本，保留上游历史、许可证、示例和[原版 README](README.upstream.md)。

## 工作流程

```text
文本 / 已有场景 → Three.js 或 Blender 场景代码
                         ↓ 按相机轨迹渲染
                     RGB proxy 视频
                         ↓
参考图 + 文本 + proxy → MiniMax-H3 Ref2VA → 生成视频（含音轨）
```

标准配置为 **960×544、124 帧、24 FPS、20 步、seed 42**；proxy 在前 **5** 步参与条件控制。相机运动通过 proxy 间接约束，标准入口没有相机矩阵或区域 mask 参数。`keep_steps` 不是像素锁定约束；部分场景和重访实验也不代表已经证明未知区域补全或几何一致性。

## 快速导航

| 内容 | 入口 |
| --- | --- |
| DCU 推理与内存管理 | [`run_dcu.py`](run_dcu.py) |
| DCU 注意力适配 | [`cvm/dcu_runtime.py`](cvm/dcu_runtime.py) |
| 标准输入和参数 | [`inference/generate_video.py`](inference/generate_video.py) |
| Three.js 渲染 | [`inference/render.py`](inference/render.py) |
| 原服务器的运行与验证记录 | [`docs/DEPLOYMENT_ZH.md`](docs/DEPLOYMENT_ZH.md) |
| 教室 / Cabazon 实验源码快照 | [`cases/`](cases/) · [使用说明](docs/EXPERIMENTS_ZH.md) |
| 导出文件清单与 SHA256 | [`docs/source_snapshot.json`](docs/source_snapshot.json) |
| CUDA 安装、场景生成、参考图与评估 | [`README.upstream.md`](README.upstream.md) |

## 环境要求

以下为已验证的 SCNet 环境，并非通用最低配置：

| 项目 | 已验证配置 |
| --- | --- |
| 系统 / 加速器 | Linux，海光 BW / gfx936，2 张约 64 GiB DCU |
| Python / PyTorch | Python 3.12，海光 PyTorch 2.7.1 |
| DTK / HIP | DTK 26.04 / HIP 6.3.26093 |
| 注意力内核 | 与上述 DTK/PyTorch 匹配的海光 `flash_attn` |
| 应用依赖 | [`requirements.txt`](requirements.txt)，含固定 Diffusers Git 提交 |
| 主机内存 | 正式样例观察到峰值 RSS 约 243.57 GiB；原服务器约 1 TiB RAM |
| 权重磁盘 | 本次 Ref2VA 必要文件约 144.05 GB，下载临时文件另需空间 |
| 媒体工具 | FFmpeg / FFprobe 可从 `PATH` 调用；原环境为 7.0.2 |
| 可选渲染 | Playwright + Chromium；无 GPU 浏览器可使用 SwiftShader |

DCU 环境需先具备供应商提供的 PyTorch 和 FlashAttention。**不要在 DCU 环境执行上游 README 的 CUDA wheel 安装命令。** 仓库不打包供应商运行时、虚拟环境、浏览器或 FFmpeg 二进制。

在已有且匹配的 DCU Python 环境中安装应用依赖：

```bash
# 确认使用海光构建
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count())'
python -c 'from flash_attn import flash_attn_func; print("flash_attn available")'

# 保留现有 torch / torchvision / torchaudio
grep -vE '^(torch|torchvision|torchaudio)==' requirements.txt > /tmp/cvm-app-requirements.txt
python -m pip install -r /tmp/cvm-app-requirements.txt
python -m pip check

# 需要 Three.js 渲染时安装
python -m playwright install chromium
ffmpeg -version
ffprobe -version
```

该片段以已有兼容 DCU 环境为前提，尚未在全新服务器上重新验证。原环境继承关系和安装版本见[部署记录](docs/DEPLOYMENT_ZH.md)。

## 权重准备

使用固定版本及哈希清单 [`configs/h3_checkpoint.json`](configs/h3_checkpoint.json)：

```bash
python download_h3.py --output models/MiniMax-H3-ref2va
```

`download_h3.py` 是上游标准入口，会下载完整清单（约 210 GB，包含本次 Ref2VA 未使用的纯文生视频 Transformer）；准备磁盘时应按该容量及临时空间估算。上表的 144.05 GB 对应历史 SCNet 最小下载集。`download_cvm.py`、`finish_cvm.py` 是这套历史下载工具，含服务器绝对路径、loopback 代理端口及并发策略，新机器使用前需调整。`status_cvm.py` 依赖历史下载器的 manifest；没有 manifest 时不能据其结果判断权重是否完整。

## DCU 快速开始

以下命令在仓库根目录执行。使用现成官方输入，不需要调用场景或参考图生成 API。

```bash
# 先激活已有 DCU 环境，并按本机位置加载 DTK
source /opt/dtk-26.04/env.sh
export HIP_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# 模型文件齐备后可离线运行
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python run_dcu.py \
  --code-video examples/robot_navigation/code_video.mp4 \
  --reference-image examples/robot_navigation/reference.png \
  --prompt-file examples/robot_navigation/prompt.txt \
  --model-root models/MiniMax-H3-ref2va \
  --text-device cuda:1 \
  --generation-device cuda:0 \
  --keep-steps 5 \
  --seed 42 \
  --output outputs/robot_navigation_dcu.mp4
```

`cuda:0` / `cuda:1` 是 PyTorch HIP 后端沿用的命名，对应 `HIP_VISIBLE_DEVICES` 暴露后的逻辑编号。需按集群规则预先分配设备，上述命令不会申请设备锁。输出文件存在时会拒绝覆盖，请改用新文件名。

正常完成后生成视频及同名 `.run.json`，并输出 `RUN_COMPLETE_SECONDS`。JSON 包含参数、耗时、媒体信息、SHA256 和各卡峰值分配显存。仅看到 MP4 不等于进程正常结束。

`run_dcu.py` 使用 Linux 信号及 HIP 环境，Windows 副本用于编辑与上传。标准 CLI 未开放步数、尺寸和帧数参数；调整前需检查 `cvm` 常量和媒体校验。

## 验证与渲染

以下检查会实际使用 DCU，请在分配好的设备上执行：

```bash
mkdir -p outputs
python preflight_dcu.py       # 小型随机模型、注意力、offload、潜变量检查
python probe_gqa.py           # 分组查询注意力数值检查
python probe_cleanup.py       # 复现旧问题并验证 DcuH3.close()
```

小型随机模型测试只验证算子和生命周期，不代表预训练模型效果。重新渲染上游 Three.js 椅子场景：

```bash
# 默认浏览器不可用时，可指定已安装的 Chromium
# export CVM_CHROMIUM_EXECUTABLE=/path/to/chromium
python render_smoke.py
```

输出 `outputs/chair_render/case/reconstruction.mp4` 仅是 proxy。自己准备的 Blender 或 Three.js proxy 可传给 `--code-video`，但须满足标准尺寸、帧数、帧率及编码检查。

## 已验证结果与边界

2026-10-27 至 2026-10-28，官方 `robot_navigation` 样例以原始非量化 Ref2VA 权重完成 20 步生成；输出 **960×544 / 124 帧 / 24 FPS / 含音轨**，全片解码核验通过。

| 阶段 | 实测耗时 |
| --- | --- |
| 模型加载与锁页准备 | 431.48 秒 |
| 20 步去噪 | 892.83 秒 |
| 启动至视频落盘 | 约 1501 秒（25 分钟） |
| 独立 Three.js 椅子场景渲染 | 387.73 秒 |

该次正式生成在视频落盘后发生退出清理异常。后续修复通过了专项小模型生命周期验证，但没有为修复重新跑完整 20 步。因此，上表不是正常退出总耗时，不含下载或 Agent 输入制作；两个不同场景的渲染和生成耗时不能直接相加。近期实验完成状态需逐项核查，源码快照本身不构成结果验证。

## 适配内容

- `cvm/dcu_runtime.py`：进程内桥接海光 FlashAttention，处理共享存储的非连续 Q/K/V，对不支持的布局回退 math SDPA。
- `run_dcu.py`：原权重锁页准备，Transformer 两个 block 一组 CPU offload，文本编码器逐模块卸载；增加有限值检查、分阶段日志和摘要。
- `DcuH3.close()`：在 inference mode 中移除 Accelerate hooks 并恢复 CPU 权重，避免 meta / inference tensor 清理异常。
- `inference/render.py`：支持 Chromium 路径、轮询场景 API 和浏览器错误诊断，适配无头渲染。

## 实验快照与仓库内容

`cases/` 收录 2026-11-19 导出时 93 个实验目录的文本源码和配置，包括部分教室、重访、多参考、mask 尝试和累计 360° 实验。它们包含失败尝试和准备中的任务；部分脚本会动态修改标准推理行为。请先阅读[实验说明](docs/EXPERIMENTS_ZH.md)。

仓库不包含模型权重、新实验 PNG / MP4 输入输出、完整运行日志或大型工具二进制。上游原有的小型示例资源保留。`docs/source_snapshot.json` 记录导出文本的 SHA256，供与服务器核对；服务器路径仅描述原环境。

## 许可证与引用

代码沿用上游 [Apache-2.0](LICENSE)；示例和第三方素材遵循 [`examples/LICENSES/`](examples/LICENSES/) 内各自许可及署名要求。MiniMax-H3 权重使用其独立许可证，不随仓库重新分发。论文引用见 [README.upstream.md](README.upstream.md#citation)。
