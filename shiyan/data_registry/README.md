# data_registry

这里记录数据指纹、审计结果、标签版本和划分版本。原始数据本体不进入 Git。

目录：

- `manifests/`：文件清单，例如路径、大小、mtime、哈希。
- `fingerprints/`：数据集、标签、split 的指纹。
- `audits/`：数据审计报告和机器可读表格。
- `label_versions/`：标签修正版本说明。
- `split_assignments/`：冻结后的 train/val/holdout 划分清单。

当前人工审计：

- `audits/v0_original/`：原始数据机器审计，保持只读。
- `audits/v1_targeted_review/`：基于验证集错误分析的第一轮 30 张重点问题图人工审计。
- `label_versions/v1_audited/`：预留的修订标签版本目录，只有在人工审计确认后才生成。
