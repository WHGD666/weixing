# v1_audited

这是后续人工审计后的标签版本预留目录。当前已有本地工作副本 `shiyan/data2`，其候选版本名为 `v1_manual_revision_candidate`；它尚未通过机器验收和重新训练，不代表 `v1_audited` 已冻结。第一轮原始数据基线不使用本目录。

当前候选范围为前 1360 张图片，主要修改舰船和 FSC/导弹车标签。`0581`、`0582`、`0583`、`0910` 经人工确认是空白标注图，训练前需要创建 4 个空的同名 `.txt` 文件。完整准备记录见 `shiyan/experiments/notes/AUDIT003_data2_manual_revision_prep.md`。

生成条件：

1. `shiyan/data_registry/audits/v1_targeted_review/review_queue.csv` 的 30 张图全部完成审核；
2. 审计结论明确区分了模型错误和标签错误；
3. 已形成修改清单，能够逐项说明原始标签、新标签和修改原因；
4. 复制原始标签生成新版本，并重新运行机器审计；
5. 新版本通过类别、框格式、文件覆盖率和 split 一致性检查。

禁止直接修改 `shiyan/data/labels/` 中的原始标签。若本轮没有足够证据证明标签错误，本目录可以继续保持只有本说明文件，不应为了训练而强行生成修订版。
