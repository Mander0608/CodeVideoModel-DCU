# Infinigen → CVM 124 帧实验

- proxy：原 24 fps 帧序列第 49–172 帧，连续截取，960×544，124 帧，约 5.17 秒。
- 路径：餐厅走向客厅；这段不是完整闭环回访。
- reference：相同场景、相机和材质，256 samples、1708×960 渲染，再按与 proxy 相同的画面范围裁切到 1920×1088。是合成渲染，不是真实照片；提高采样主要减少噪声。
- CVM：MiniMax-H3 Ref2VA，seed42，20 步，前 5 步使用 proxy。参考图贯穿生成。沿用已有 adapter，在进程内取消“只复制参考图风格，不复制布局”的自动附加语句；共享项目不修改。
- 相机仅通过 RGB proxy 间接约束，生成结果不保证逐像素/逐位姿一致。
- inputs/：proxy、对应 proxy 首帧、最终参考图、文本。
- camera_trajectory.npz：与 124 帧及裁切缩放对应的相机数据，仅作记录，不作为 CVM 输入。
- input_manifest.json、reference_provenance.json：来源与校验。
- outputs/：实际模型输出；timing.json、run.log：运行记录。
