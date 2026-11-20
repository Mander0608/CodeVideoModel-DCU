# 实验源码快照使用说明

`cases/` 来自 2026-11-19 导出的 SCNet 工作目录，包含 93 个 case。它不是实时同步，也不是全部实验成功的声明。

## 内容与依赖

- `cvm_config.json`：输入路径、seed、步数、proxy 保留步数、逻辑设备和输出路径。
- `run_cvm.py` / `run_case.py`：启动适配。不同实验可能删除默认风格提示词、加入多参考或 mask 逻辑，需逐个阅读。
- `preflight.py`：该实验的输入或设备检查。
- `launch.sh` / `supervise.py`：历史启动、排队或监督逻辑，可能含绝对路径和设备编号。
- manifest / checksum / timing / status 等 JSON：导出时的记录，不应视为当前状态。

视频、PNG 参考图、mask、权重和大型产物没有导入 Git。文本目录可能保留 `inputs/prompt.txt` 等，但这不意味着输入齐备。运行前应恢复完整输入，并核对配置和清单。部分 case 依赖仓库以外的 Blender 场景维护脚本或其他项目；本快照不包含完整 Code-as-Memory 编排系统。

两个 `classroom_revisit8_ref3_*` case 的 `provenance/layout_source.py`、`presentation_source.py` 是保留缩进的类方法摘录，用来记录当时依赖实现，不能作为独立 Python 模块执行。源码哈希记录针对导出时的原始字节；Git 文本换行归一化可能使克隆后的文件字节不同。原始格式和历史记录中的空白不作为推理代码修改处理。

## 阅读一个累计教室实验

以 `cases/classroom360_cumulative_r09_20261119_121621/` 为例：

1. 查看 `cvm_config.json` 和 `input_manifest.json`，恢复 `inputs/partial_proxy.mp4`、`inputs/reference.png`、`inputs/prompt.txt`。
2. 核对逻辑设备、启动器物理设备和绝对路径。
3. 输入齐备后，仅打印推理命令：

```bash
python cases/classroom360_cumulative_r09_20261119_121621/run_cvm.py \
  --project-root "$PWD"
```

该适配器默认 dry run，但打印前仍会断言输入存在。分配好设备并准备环境、模型后，才可加 `--execute` 运行。该 case 会去掉标准 CVM 自动追加的风格引导文本，和标准入口的提示词行为不同。

## 解释结果

RGB proxy 约束不等于 mask inpainting，也不保证请求相机与生成相机完全一致。case 名称中的 `masked`、`multiref`、`seam` 等是实验变体名称，不能据此推断标准入口支持相应功能或结果已验证。

评估时应同时检查实际视频、退出状态、有效提示词、输入校验和及日志。仅存在配置、启动标记或历史 JSON，不能证明生成成功。
