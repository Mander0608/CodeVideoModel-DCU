# Code Video Model / scnet2 部署记录

开始：2026-10-27；运行验证跨至 2026-10-28（北京时间）。

## 最终结果

**已在 scnet2 生成官方尺寸和采样配置的视频，并完整解码检查通过。** 输出为 `outputs/robot_navigation_dcu.mp4`：960×544、24 FPS、124 帧（约 5.17 秒），包含音轨，20 次更新、前 5 步代理条件，使用完整原始权重、不量化。

需要区分输出成功和进程正常退出：这次运行在视频导出、校验和最终改名后，退出清理因 inference-mode 参数恢复而报错。因此保留失败日志，没有补造 `RUN_COMPLETE_SECONDS` 或正常完成的 `.run.json`。最终修复在 `DcuH3.close()` 添加 `@torch.inference_mode()`，专项验证见 `cleanup_validation.json`；未为这一清理改动重新执行整段 20 步生成。

本地结果与核验材料：

- `robot_navigation_dcu.mp4`：实际 H3 生成视频。
- `robot_navigation_comparison.png`：代理/生成结果在第 0、31、62、93、123 帧的对照。
- `robot_navigation_media_validation.json`：全片解码、尺寸、24 FPS、124 帧和音轨检查。
- `generation-full.log`：20 步实际执行记录、导出及清理异常的原始日志。
- `result_summary.json`：如实区分生成成功、清理错误和后续修复的摘要。

实测耗时：从正式启动到视频文件落盘约 **25 分钟**（含 Python 导入、加载、锁页准备、编码、采样和导出）；其中模型加载/锁页约 **431.48 秒**，20 步去噪合计 **892.83 秒**。前五步约 51–61 秒/步，后十五步约 37–46 秒/步。此处总耗时由启动日志与输出文件时间估算，不含之前的下载和失败尝试。

抽帧观察：机器人在五个采样时刻持续存在，位置从左侧移动至右侧出口附近，基本跟随代理的大体路线；生成画面的材质和照明更接近参考图。此为定性检查，未计算轨迹/接触误差、未逐像素对齐，也没有复现整个 CodeVideoBench 的评分。对照图本身不能证明全片不存在短暂伪影。

## 环境与目录

- 服务器：`scnet2`，主机 `m09r2n04`。
- 代码：`/public/home/lixin1/chenxy/CodeVideoModel`。
- 独立环境：`/public/home/lixin1/chenxy/envs/cvm-dcu`。
- 启动器：`/public/home/lixin1/chenxy/run_cvm_dcu.sh`。
- 硬件：海光 BW / gfx936，单卡约 64 GiB；预定使用物理 DCU 0、1。
- 软件：DTK 26.04、HIP 6.3.26093、PyTorch 2.7.1、Python 3.12；Diffusers 固定官方指定 Git 提交，Transformers 5.12.1、Accelerate 1.14.0、PEFT 0.19.1。
- 环境通过 `venv --system-site-packages` 只读继承 `/public/home/lixin1/envs/sam3.1-dtk2604-py312` 的海光 PyTorch 等依赖；该基础环境需要保留。新安装包在 `cvm-dcu` 内，不修改其他实验环境。

## 已完成的实际检查

| 项目 | 实际结果 | 证据 |
|---|---|---|
| Python 依赖一致性 | `pip check` 通过 | 终端安装记录、`environment-freeze.txt` |
| DCU 注意力 | 常规独立 Q/K/V 最大绝对误差约 0.004；修复共享非连续视图错误后，压力样例误差 0.03125 | `attention-probe.log`、`preflight.json` |
| H3 小型网络前向 | 有限的视频/音频预测输出 | `preflight.json` |
| BF16 / 指定 FP32 模块的逐块卸载 | 与同一小型网络常驻 DCU 的结果完全一致 | `group_offload_max_abs_errors=[0.0,0.0]` |
| Qwen 分组查询注意力 | 129 / 1024 token、因果 / 非因果均通过，最大绝对误差 ≤0.03125 | `gqa.log` |
| 视频潜变量打包、还原 | 通过精确往返检查 | `preflight.json` |
| Three.js 真实渲染 | 成功；960×544、24 FPS、124 帧，H.264 / yuv420p，无音轨 | `chair_proxy_scnet2.mp4`、`render_report.json` |
| 渲染耗时 | 387.73 秒 | 包含初始检查、124 帧导出及预览图，CPU SwiftShader |
| 预训练 H3 缩小测试 | 已完成两步去噪、解码、带音轨视频导出；随后清理报错 | `generation-smoke-cleanup-error.log`、`robot_navigation_smoke_dcu.mp4`、`smoke_media.json` |
| 正式配置 | 完成 20 步、解码和视频导出；随后清理异常 | `generation-full.log`、`robot_navigation_dcu.mp4` |
| 最终清理修复 | 在 inference mode 内恢复 CPU 参数再释放 | 见 `cleanup-validation.log`、`cleanup_validation.json` |
| 正式输出核验 | 全片解码无错误，960×544、24 FPS、124 帧、含音轨，本地与服务器 SHA256 相同 | `robot_navigation_media_validation.json` |
| 必要权重下载 | 61 个文件全部完成官方 SHA256 / 索引校验，总计 144.05 GB | `download-ranges.log`、`download-manifest.json` |

**小型网络测试采用随机参数，只证明相关 DCU 算子和卸载路径可执行，不是预训练模型效果复现。Three.js 代理视频也不是最终 H3 生成视频。**

## 实现适配

1. 通过 `cvm/dcu_runtime.py` 在当前进程接入海光 FlashAttention，保留不支持布局的 PyTorch math fallback。实测发现共享存储的非连续 Q/K/V 会发生明显数值错误，已显式转为连续布局。
2. `run_dcu.py` 在官方流程外添加内存管理：H3 transformer 每次搬运两个 block，Qwen3-VL 用 Accelerate 逐模块卸载。原始权重不量化，正式配置保留 20 次更新和前 5 步代理条件。
3. 真实运行发现文件映射权重直接传输到 HIP 设备很慢，堆栈定位在模型 `.to()`。加入一次性主机锁页内存准备，保持原始 dtype 和数值。10.49 MB 的独立传输探测中，映射内存 H2D 为 0.152 秒，匿名副本为 0.00142 秒，锁页副本为 0.00121 秒；后两者另有 0.0966 / 0.0126 秒准备成本。这些是小块传输测量，不能等同于完整推理加速比。记录见 `transfer.log`。
4. `inference/render.py` 支持指定 Chromium 路径，并将等待场景 API 的机制改为定时轮询，解决此环境中基于 requestAnimationFrame 的等待超时；加入浏览器错误诊断。
5. FFmpeg / FFprobe 7.0.2 静态程序放在项目 `tools/ffmpeg/`，没有替换系统程序。
6. `download_cvm.py` / `finish_cvm.py` 按官方固定 revision 下载需要的 61 个文件，合计 144,051,157,679 字节。大文件分段断点续传，校验响应的 Content-Range，并按官方 SHA256 验证合并后的文件。未下载本次未使用的纯文生视频 transformer。
7. Accelerate CPU offload 在前向后保留 meta 占位参数，官方 `ComponentsManager.remove()` 会对它们调用 `.to('cpu')` 并报错。`DcuH3.close()` 在 `torch.inference_mode()` 内先调用 `remove_hook_from_module(..., recurse=True)` 恢复原 CPU 权重，再执行官方清理。第一次专项检查没有覆盖 inference tensor，故未发现第二个边界问题；最终 `probe_cleanup.py` 按实际推理上下文重建该问题，先要求旧行为报出同样错误，再验证实际 `DcuH3.close()`。

锁页权重本身约 144 GB，连同运行时副本、工作区等，解码/导出附近观察到的进程峰值 RSS 达约 **243.57 GiB**（`VmHWM=255399348 kB`，见 `full-memory-snapshot.log`）；这是采样到的峰值下界，不是进程退出后的最终测量。不能仅依据“两张 64 GiB 卡”判断其他机器是否足够。本机有约 1 TiB RAM，允许本次分配。清理异常发生在正常汇总前，因此未获得最终各卡峰值显存；日志中的 `allocated_GiB≈9.96` 仅是每步完成时分配量，不能当作峰值。

缩小测试输出为 256×160、24 FPS、22 帧，音轨为 32 kHz 双声道 AAC。已抽帧检查，画面存在明显纹理伪影；两步测试不用于评价最终画质。视频已经通过格式检查并从临时文件改名，随后资源清理异常导致进程非零退出，所以保留原始失败日志，不为该次补造正常完成的 `.run.json`。

## 运行与检查命令

查看权重进度：

```bash
ssh scnet2 'python3 /public/home/lixin1/chenxy/CodeVideoModel/status_cvm.py'
```

查看推理日志：

```bash
ssh scnet2 'tail -c 3000 /public/home/lixin1/chenxy/CodeVideoModel/logs/generation-full.log'
```

正式配置（输出文件已存在时会拒绝覆盖，请改名）：

```bash
ssh scnet2
bash /public/home/lixin1/chenxy/run_cvm_dcu.sh run_dcu.py \
  --code-video examples/robot_navigation/code_video.mp4 \
  --reference-image examples/robot_navigation/reference.png \
  --prompt-file examples/robot_navigation/prompt.txt \
  --model-root models/MiniMax-H3-ref2va \
  --keep-steps 5 \
  --output outputs/robot_navigation_dcu.mp4
```

本次排队脚本：`/public/home/lixin1/chenxy/queue_cvm.sh`。它先等全部必要文件校验完成，再检查算子预检结果，申请与已有 WorldCrafter 启动器一致的设备文件锁，确认两张卡可用，然后执行：

- 缩小测试：256×160、22 帧、2 步，前 1 步代理条件，参考图短边降至 512；输出 `outputs/robot_navigation_smoke_dcu.mp4`。
- 正式测试：960×544、124 帧、20 步，前 5 步代理条件；输出 `outputs/robot_navigation_dcu.mp4`。

以后正常完成时应出现 `RUN_COMPLETE_SECONDS`、视频通过流格式检查并生成对应 `.run.json`。本次正式视频已通过独立核验，但清理异常使进程未正常退出，详细边界见本文“最终结果”。低分辨率测试用于排错，不用于评价论文效果。

修复清理错误后，正式运行使用 `/public/home/lixin1/chenxy/run_full_cvm.sh`，直接执行正式配置，保留已有缩小视频。它同样申请两张设备的文件锁，并检查空闲显存。

## 网络及外部模型依赖

scnet2 无法直接访问 Hugging Face 前端，本次通过 SSH 的远程 loopback 端口 `17997` 转发到本机代理，用于获取下载元数据和签名链接；大权重优先从 CDN 直连分段下载。断线重启下载时需要恢复同样的通道。离线推理本身不需要该通道。

本次使用官方现成示例，未调用付费图像生成服务，也未测试从零提示词自动编写场景。官方从零场景生成需要可用的 Copilot CLI 认证，参考图自动生成需要 `OPENROUTER_API_KEY`；这些属于后续可选输入制作步骤，不是本次 H3 示例测试的前置条件。
