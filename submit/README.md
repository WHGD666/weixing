# submit：Docker 交付版本总目录

`submit/` 只负责管理最终交付版本，不存放训练过程、验证集结果、错误分析或数据审计产物。每个版本必须使用独立目录，确认后的版本不能覆盖修改。

## 版本结构

```text
submit/
├── README.md
├── v1/                         # 第一轮原始数据基线候选
│   ├── README.md
│   ├── Dockerfile
│   ├── environment.yml
│   ├── app/
│   ├── models/
│   ├── manifests/
│   ├── test-input/
│   └── test-output/
├── v2/                         # 后续独立候选，按需创建
└── v3/                         # 后续独立候选，按需创建
```

当前 `v1` 已完成模型、推理入口、Linux 运行环境和 Docker 镜像的本地验证，等待比赛平台 ACR 推送与试运行。模型、测试图片和测试输出仍保持本地，不进入 GitHub。

## 版本规则

- `v1`：比赛方原始数据、原始标签、`EXP001` 基线，不进行人工修标；本地 Docker 候选已验证。
- 后续版本只有在模型、参数和交付内容冻结后才创建。
- 新版本复制结构，不修改已经记录过的旧版本。
- 每个版本独立保存 Dockerfile、环境、推理代码、模型、测试结果和 manifest。
- 版本目录中的模型权重和测试输入输出只用于本地构建与验证，不上传到 GitHub 源码仓库。

## 相关记录

- [v1 交付工作区](/D:/daima/weixing/submit/v1/README.md)
- [官方要求核对表](/D:/daima/weixing/shiyan/docker/官方要求核对表_封装前.md)
- [Docker 说明汇总](/D:/daima/weixing/shiyan/docker/Docker封装与评测系统说明汇总.md)
