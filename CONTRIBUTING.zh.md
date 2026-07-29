# 感谢你的关注与贡献！

[🇺🇸 English](CONTRIBUTING.md)

Autoware 由像你这样的贡献者支持，各种类型和大小的贡献都欢迎。作为贡献者，以下是我们希望你遵循的 Autoware 及其相关仓库的指南：

## 行为准则

为确保 Autoware 社区保持开放和包容，请遵守[行为准则](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/blob/main/CODE_OF_CONDUCT.md)。

如果你认为社区中有人违反了行为准则，请发送邮件至 conduct@autoware.org 进行举报。

## 开始之前需要了解什么？

### 关于本项目

请确保你已阅读本项目的 [README](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/blob/main/README.md) 文件和[上手指南](https://github.com/autowarefoundation/autoware.privately-owned-vehicles/blob/main/ONBOARDING.md)，以了解本项目的目标和发展方向。

### 参与开源项目

如果你是开源新手，我们推荐阅读 [GitHub 的如何参与开源项目](https://opensource.guide/how-to-contribute)指南，了解人们为什么参与开源项目、参与意味着什么等更多内容。

## 如何获取帮助？

请勿就一般支持问题提交 issue，因为我们希望将 GitHub issue 用于已确认的 bug 报告。相反，请在 Q&A 类别中发起讨论。有关 Autoware 支持机制的更多详情，请参阅支持指南。

## 注意

用于提问或未经确认的 bug 的 issue 将被维护者移至 GitHub 讨论区。

## 如何参与贡献？

### 讨论

你可以通过参与和促进[讨论](https://github.com/orgs/autowarefoundation/discussions)来为 Autoware 做贡献，例如：

- 提出增强 Autoware 的新功能建议
- 加入现有讨论并表达你的观点
- 为其他贡献者组织讨论
- 回答问题并支持其他贡献者

### 加入并参与私人车辆工作组

参加并参与私人车辆工作组的会议。我们每周一在两个时间段开会，讨论当前开发进展、新功能以及与本项目技术执行相关的其他重要议题。

有两个时间段可选（时段 1 更适合东亚参与者，时段 2 更适合欧洲/美国参与者）。你可以从以下链接添加会议邀请：[时段 1 会议链接](https://calendar.google.com/calendar/u/0/r/week/2024/11/18?eid=MzlmZDZvNjhjZ3FwOXZkMjc4cHZqbHBhaDhfMjAyNDExMThUMDQzMDAwWiBhdXRvd2FyZS5vcmdfNmxvbDBobzVmdDAyMTdoOGM2MHBpMWZtMzBAZw) 和 [时段 2 会议链接](https://calendar.google.com/calendar/u/0/r/week/2024/11/18?eid=aG0yMWUzNXU1N2JxYW9wMHZkb2lncmg5bGNfMjAyNDExMThUMTYwMDAwWiBhdXRvd2FyZS5vcmdfNmxvbDBobzVmdDAyMTdoOGM2MHBpMWZtMzBAZw) — 两个时段讨论相同的议题，只需根据你的时间安排参加其中一个即可。

### Bug 报告

在报告 bug 之前，请先搜索 issue 追踪器中对应的仓库。可能已经有人报告了相同的问题并提供了变通方案。如果无法确定合适的仓库，请在 [Q&A 类别](https://github.com/autowarefoundation/autoware/discussions/new?category=q-a)中发起新讨论向维护者求助。

报告 bug 时，请提供一组最小复现步骤。这样可以帮助我们快速确认并聚焦于正确的问题。

如果你想自行修复 bug，这很好，但请在提交 PR 之前先在 issue 中与维护者讨论可能的修复方案。

[创建 issue 很简单](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue#creating-an-issue-from-a-repository)，但如果遇到问题，请发起 Q&A 讨论求助。

### Pull Request

你可以就以下小型更改提交 pull request：

- 小型文档更新
- 修正拼写错误
- 修复 CI 失败
- 修复编译器或分析工具检测到的警告
- 对单个包进行小型更改

如果你的 pull request 是大型更改，应遵循以下流程：

1. [创建 GitHub 讨论](https://docs.github.com/en/discussions/collaborating-with-your-community-using-discussions/collaborating-with-maintainers-using-discussions)来提议更改。这样做可以让你从其他成员和 Autoware 维护者那里获得反馈，并确保提议的更改符合 Autoware 的设计理念和当前开发计划。如果不确定在哪里发起讨论，请[创建新的 Q&A 讨论](https://github.com/autowarefoundation/autoware/discussions/new?category=q-a)。

2. 在讨论达成共识后[创建 issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue)

3. [创建 pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)来实现更改，并引用第 2 步创建的 issue

4. 为新功能创建文档（如适用）

大型更改的示例包括：

- 为 Autoware 添加新功能
- 添加新的文档页面或章节

有关如何提交高质量 pull request 的更多信息，请阅读 [pull request 指南](https://autowarefoundation.github.io/autoware-documentation/main/contributing/pull-request-guidelines/)，别忘了查看所需的[许可证标注](https://autowarefoundation.github.io/autoware-documentation/main/contributing/license/)！

### Pull Request 检查

我们有多种 CI 工作流检查来确保 pull request 的质量。

#### `semantic-pull-request`

此工作流确保 pull request 标题遵循[约定式提交](https://www.conventionalcommits.org/en/v1.0.0/)规范。

更多详情见 [Autoware 文档](https://autowarefoundation.github.io/autoware-documentation/main/contributing/pull-request-guidelines/#apply-conventional-commits-to-the-pull-request-title-required-automated)。

#### `pre-commit`

[pre-commit](https://pre-commit.ci/) 是一个在你提交时运行格式化器或 lint 器的工具。

此工作流检查 pull request 是否没有 pre-commit 错误。

更多信息见 [Autoware 文档](https://autowarefoundation.github.io/autoware-documentation/main/contributing/pull-request-guidelines/ci-checks/#pre-commit)。
