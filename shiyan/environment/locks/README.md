# 环境锁定文件

这里保存已经验证通过的环境快照，例如：

- `weixing-pip-freeze.txt`
- `weixing-conda-list.txt`

这些文件由本机命令生成，不手工编写。每次正式实验都应该在 run manifest 中引用对应的环境快照。
