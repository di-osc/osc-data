# 项目说明

## 项目原则
- 所有的功能的实现尽量使用rust完成，尽可能的减少python依赖。

## 注释风格
每个新功能的函数都需要有完整的注释，包括参数、返回值、示例等。且示例需要完整。

## 项目发布
本项目通过github action自动发布到pypi，每次发布需要提交git tag，tag的格式为v{version}，version的格式为{major}.{minor}.{patch}，例如v0.2.8。

