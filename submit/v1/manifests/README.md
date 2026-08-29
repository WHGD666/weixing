# submit/v1/manifests

这里保存第一轮交付版本的证据记录，不保存训练数据。

最终至少记录：

- 交付版本编号；
- Git commit 和工作区状态；
- 模型文件路径与 SHA-256；
- 类别映射和 tiled 推理参数；
- `environment.yml` 来源环境；
- Docker 镜像平台、tag 和构建时间；
- `--network none` 本地运行结果；
- `result.json` 字段、图片覆盖率和坐标检查结果；
- 官方平台提交时间、tag、任务状态和返回成绩。

当前已建立并更新：

- `v1_delivery_freeze.md`：v1 阶段性冻结边界、模型哈希、推理参数和本地验证结果；
- `v1_runtime_dependency_plan.md`：Windows 研发依赖与 Docker 运行依赖的边界；
- `../../experiments/notes/INF008_docker_full_validation.md`：Docker GPU、断网和 897 张全量验证记录。

当前 manifest 记录的是本地交付候选状态。ACR push、官方平台提交时间、任务状态和封闭测试成绩，需要在比赛系统开放并返回结果后补入，不能提前填写或用本地指标替代。
