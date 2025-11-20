# 🌌 LLM 学习指南与资源导航 (LLM Learning & Resources)

一个收集大模型学习路径、开源项目、实战教程与工具集合的导航仓库。如果不嫌弃，请给我点个 Star ⭐️，这是我更新的动力！

🔍 说明:
- 排名不分先后：资源主要按类别整理。
- 持续更新：作者于 2025.7 入坑LLM，正在不断学习和补充中。
- 免责声明：内容来源于网络，若侵权请联系删除。

<details>
<summary>📅 更新日志 (Update Log)</summary>

**2025.11.20 (Third Update)**
- Refactor: 将 AntiGravity 移至工具链 IDE 板块
- New: 添加了 Coze
- Structure: 重构了目录结构，分类更清晰

**2025.11.18 (Second Update)**
- Add: 基础教程, Blogger, Blog

**2025.11.16 (First Release)**
- Init: 项目开源

</details>

## 📚 目录 (Table of Contents)

- [🌌 LLM 学习指南与资源导航 (LLM Learning \& Resources)](#-llm-学习指南与资源导航-llm-learning--resources)
  - [📚 目录 (Table of Contents)](#-目录-table-of-contents)
  - [1. 🛣️ 基础与理论 (Foundations)](#1-️-基础与理论-foundations)
    - [数学、机器学习、深度学习入门](#数学机器学习深度学习入门)
    - [LLM 理论与综述](#llm-理论与综述)
  - [2. ⚔️ 核心实战与复现 (Core Implementation)](#2-️-核心实战与复现-core-implementation)
    - [从零手写/复现 LLM](#从零手写复现-llm)
    - [模型微调 (Fine-tuning)](#模型微调-fine-tuning)
    - [开发者教程与手册](#开发者教程与手册)
  - [3. 🏗️ 应用开发架构 (Application Engineering)](#3-️-应用开发架构-application-engineering)
    - [Agent (智能体)](#agent-智能体)
    - [RAG (检索增强生成)](#rag-检索增强生成)
    - [应用开发实战](#应用开发实战)
  - [4. 🧩 垂直领域与多模态 (Vertical \& Multimodal)](#4--垂直领域与多模态-vertical--multimodal)
    - [多模态 (CV/Audio)](#多模态-cvaudio)
    - [垂直行业模型](#垂直行业模型)
  - [5. 🛠️ 工具链与生态 (Tools \& Ecosystem)](#5-️-工具链与生态-tools--ecosystem)
    - [核心框架与官网](#核心框架与官网)
    - [IDE 与开发工具](#ide-与开发工具)
    - [算力与炼丹平台](#算力与炼丹平台)
    - [API 服务与聚合](#api-服务与聚合)
  - [6. 🌐 社区与资讯 (Community \& News)](#6--社区与资讯-community--news)
    - [博主](#博主)
    - [博客](#博客)
  - [7. 🎮 DLC](#7--dlc)
    - [算法](#算法)
    - [Others](#others)

---

## 1. 🛣️ 基础与理论 (Foundations)


### 数学、机器学习、深度学习入门

| 资源名称 | 作者/组织 | 链接 | 备注 |
|---------|----------|------|------|
| ML-For-Beginners | Microsoft | [GitHub](https://github.com/microsoft/ML-For-Beginners) | ML 书籍 |
| 动手学深度学习 2.0 | - | [在线阅读](https://zh.d2l.ai/chapter_preface/index.html) | 深度学习经典教材 |
| machine_learning_notebook | 583 | [GitHub](https://github.com/583/machine_learning_notebook) | 机器学习笔记 |
| CS224n-Reading-Notes | LooperXX | [GitHub](https://github.com/LooperXX/CS224n-Reading-Notes) | 斯坦福 CS224n (NLP) 课程笔记 |
| Book-Math-Foundation-of-RL | MathFoundationRL | [GitHub](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning) | 强化学习的数学基础 |
| stanford-cs336-a1 | Spectual | [GitHub](https://github.com/Spectual/stanford-cs336-a1) | 斯坦福 CS336 作业 |

### LLM 理论与综述

| 资源名称 | 作者/组织 | 链接 | 备注 |
|---------|----------|------|------|
| Foundations-of-LLMs | ZJU-LLMs | [GitHub](https://github.com/ZJU-LLMs/Foundations-of-LLMs) | 大模型基础电子书 |
| Foundations of LLMs (Paper) | 肖桐老师 | [论文](https://arxiv.org/abs/2501.09223) | 论文搬运，大模型基础综述 |
| so-large-lm | Datawhale | [GitHub](https://github.com/datawhalechina/so-large-lm) | 大模型基础知识梳理 |
| 大模型快速入门学习路径 | - | [知乎文章](https://zhuanlan.zhihu.com/p/685915213) | 知乎文章 |
| 3万star的LLM公开资料 | - | [知乎文章](https://zhuanlan.zhihu.com/p/686277638) | 大模型入门教程合集 |
| AI-Resources-Central | CoderSJX | [GitHub](https://github.com/CoderSJX/AI-Resources-Central) | 全球优秀 AI 开源项目汇总 |
| study-progress-of-llm | mikelikeai | [GitHub](https://github.com/mikelikeai/study-progress-of-llm) | 个人 LLM 学习过程总结 |
| Agentic Design Patterns | ginobefun | [GitHub](https://github.com/ginobefun/agentic-design-patterns-cn) | Agent 设计模式中文翻译 |

---

## 2. ⚔️ 核心实战与复现 (Core Implementation)


### 从零手写/复现 LLM

| 资源名称 | 作者/组织 | 链接 | 备注 |
|---------|----------|------|------|
| MiniMind | jingyaogong | [GitHub](https://github.com/jingyaogong/minimind) | 🔥 大模型全阶段复现，极佳的入门教程 |
| MiniMind-in-Depth | hans0809 | [GitHub](https://github.com/hans0809/MiniMind-in-Depth) | MiniMind 的详细解析教程 |
| 【2025/Minimind】Only三小时！Pytorch从零手敲大模型，架构到训练全教程 | 木乔_Mokio | [B站](https://www.bilibili.com/video/BV1T2k6BaEeC?spm_id_from=333.788.videopod.episodes&vd_source=3151b98d67ade6395736508def783435) | 手敲MiniMind |
| transformers-code | zyds | [GitHub](https://github.com/zyds/transformers-code) | 手把手带你实战 Transformers 代码 |
| Hands-On-LLMs-CN | bruceyuan | [GitHub](https://github.com/bbruceyuan/Hands-On-Large-Language-Models-CN) | 《动手学习大模型》中文版 |
| llm-course | mlabonne | [GitHub](https://github.com/mlabonne/llm-course) | 系统性的 LLM 课程 |

### 模型微调 (Fine-tuning)

| 资源名称 | 作者/组织 | 链接 | 备注 |
|---------|----------|------|------|
| LLaMA-Factory | hiyouga | [GitHub](https://github.com/hiyouga/LLaMA-Factory) | 零代码/WebUI 微调百余种大模型，强烈推荐 |
| self-llm | Datawhale | [GitHub](https://github.com/datawhalechina/self-llm) | 开源大模型食用(部署/微调)指南 |
| Awesome-Chinese-LLM | HqWu-HITCS | [GitHub](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM) | 中文大模型、微调及数据集整理 |

### 开发者教程与手册

| 资源名称 | 作者/组织 | 链接 | 备注 |
|---------|----------|------|------|
| llm-universe | Datawhale | [GitHub](https://github.com/datawhalechina/llm-universe) | 面向小白开发者的大模型应用开发教程 |
| llm-cookbook | Datawhale | [GitHub](https://github.com/datawhalechina/llm-cookbook) | 面向开发者的大模型手册 |
| tiny-universe | Datawhale | [GitHub](https://github.com/datawhalechina/tiny-universe) | 大模型白盒子构建指南 |
| happy-llm | Datawhale | [GitHub](https://github.com/datawhalechina/happy-llm) | 从零开始的原理与实践 |
| ModelScope教程 | ModelScope | [GitHub](https://github.com/modelscope/modelscope-classroom) | 魔搭社区深度学习教程 |

---

## 3. 🏗️ 应用开发架构 (Application Engineering)


### Agent (智能体)

| 资源名称 | 作者/组织 | 链接 | 备注 |
|---------|----------|------|------|
| OpenManus | FoundationAgents | [GitHub](https://github.com/FoundationAgents/OpenManus) | 开源版 Manus，无需邀请码实现想法 |
| Coze | - | [文档](https://www.coze.com/open/docs/zh_cn/wel%20come.html) | 无代码搭建 Agent 平台 |
| Langchain-Chat | chatcat-space | [GitHub](https://github.com/chatchat-space/Langchain-Chatchat) | 经典的本地离线 RAG 与 Agent 框架 |
| NagaAgent | xxiii8322766509 | [GitHub](https://github.com/Xxiii8322766509/NagaAgent) | 功能丰富的智能对话助手系统 |
| Kimi CLI | MoonshotAI | [GitHub](https://github.com/MoonshotAI/kimi-cli?tab=readme-ov-file) | Kimi 自研命令行智能体工具 |


### RAG (检索增强生成)

| 资源名称 | 作者/组织 | 链接 | 备注 |
|---------|----------|------|------|
| RAGAS | explodinggradients | [GitHub](https://github.com/explodinggradients/ragas) | RAG 系统的评估与测评框架 |
| 2024年RAG 技术重大突破：一文速览全年RAG 技术革新与里程碑 | 杨夕 | [知乎](https://www.zhihu.com/question/642650878/answer/86323321960) | 2024年RAG 技术 | 

### 应用开发实战

| 资源名称 | 作者/组织 | 链接 | 备注 |
|---------|----------|------|------|
| thenextagent | qingningLime | [GitHub](https://github.com/qingningLime/thenextagent) | 基于 Qwen-VL 的自动化电脑操作工具 |
| DM-Code-Agent | hwfengcs | [GitHub](https://github.com/hwfengcs/DM-Code-Agent) | 专注于软件开发的 Code Agent |
| ai-app | GuoCoder | [GitHub](https://github.com/GuoCoder/ai-app) | AI 大模型应用集合 |
| deepseek-Lunasia-2.0 | 1112021 | [GitHub](https://github.com/1112021/deepseek-Lunasia-2.0) | 智能桌面 AI 助手 |
| tomori-chatbot | Shenyqqq | [GitHub](https://github.com/Shenyqqq/tomori-chatbot) | 高松灯聊天机器人 (趣味应用) |

---

## 4. 🧩 垂直领域与多模态 (Vertical & Multimodal)

特定领域的解决方案与视觉/音频模型。

### 多模态 (CV/Audio)

| 资源名称 | 作者/组织 | 链接 | 备注 |
|---------|----------|------|------|
| GPT-SoVITS | RVC-Boss | [GitHub](https://github.com/RVC-Boss/GPT-SoVITS) | 强大的 AI 变音与语音合成工具 |
| DeepSeek-OCR | DeepSeek | [GitHub](https://github.com/deepseek-ai/DeepSeek-OCR) | 深度求索开源的 OCR 模型 |
| awesome-pretrained-chinese | lonePatient | [GitHub](https://github.com/lonePatient/awesome-pretrained-chinese-nlp-models) | 中文预训练模型/多模态模型集合 |

### 垂直行业模型

| 资源名称 | 领域 | 链接 | 备注 |
|---------|------|------|------|
| DISC-LawLLM | ⚖️ 法律 | [GitHub](https://github.com/FudanDISC/DISC-LawLLM) | 法律领域大模型 |
| DoctorGLM | 💊 医疗 | [GitHub](https://github.com/xionghonglin/DoctorGLM) | 基于 ChatGLM-6B 的中文问诊模型 |

---

## 5. 🛠️ 工具链与生态 (Tools & Ecosystem)


### 核心框架与官网

| 名称 | 类别 | 链接 |
|------|------|------|
| PyTorch | 深度学习框架 | [官网](https://pytorch.org) |
| TensorFlow | 深度学习框架 | [官网](https://www.tensorflow.org) |
| HuggingFace | 模型库 | [官网](https://huggingface.co) |
| ModelScope | 模型库(国内) | [官网](https://modelscope.cn) |
| LangChain | 开发框架 | [官网](https://www.langchain.com) |
| OpenAI | 模型厂商 | [官网](https://openai.com) |

### IDE 与开发工具

| 资源名称 | 厂商/作者 | 链接 | 备注 |
|---------|----------|------|------|
| AntiGravity | Google | [访问地址](https://antigravity.google/) | Google Agent 编程 IDE |
| Cursor | cursor | [访问地址](https://cursor.com/cn) | IDE |
| 通义灵码 | 通义 | [官网](https://lingma.aliyun.com/lingma/) | 智能开发工具 |

### 算力与炼丹平台

| 平台名称 | 链接 | 备注 |
|---------|------|------|
| AutoDL | [官网](https://www.autodl.com/home) | - |
| OpenBayes | [官网](https://openbayes.com) | - |
| 阿里云百炼 | [官网](https://www.aliyun.com/) | 阿里大模型服务平台 |
| PPIO | [官网](https://ppio.com/user/register?from=ppinfra&invited_by=OCPKCN&utm_source=github_openmanus&utm_medium=github_readme&utm_campaign=link) | 智谱 |

### API 服务与聚合

| 平台名称 | 链接 | 备注 |
|---------|------|------|
| DeepSeek 开放平台 | [平台](https://platform.deepseek.com/usage) | 官方 API |
| 阿里云 Model Studio | [平台](https://help.aliyun.com/zh/model-studio/get-api-key) | 阿里 API |
| 娜迦 API | [平台](https://naga.furina.chat/workspace) | 第三方聚合 |

---

## 6. 🌐 社区与资讯 (Community & News)

### 博主
| 平台 | 博主 | 链接 | 备注 |
|------|------|------|------|
| B站 | 东川路第一可爱猫猫虫 | [空间](https://space.bilibili.com/675505667?spm_id_from=333.1387.follow.user_card.click) | - |
| B站 | happy魇 | [空间](https://space.bilibili.com/478929155?spm_id_from=333.1387.follow.user_card.click) | - |
| B站 | 偷星九月333 | [空间](https://space.bilibili.com/349950942?spm_id_from=333.1387.follow.user_card.click) | - |
| B站 | 堂吉诃德拉曼查的英豪 | [空间](https://space.bilibili.com/341376543?spm_id_from=333.1387.follow.user_card.click) | - |
| B站 | 你可是处女座啊 | [空间](https://space.bilibili.com/21060026?spm_id_from=333.1387.follow.user_card.click) | - |
| B站 | chaofa用代码打点酱油 | [空间](https://space.bilibili.com/12420432?spm_id_from=333.1387.follow.user_card.click) | - |
| B站 | 马克的技术工作坊 | [空间](https://space.bilibili.com/1815948385?spm_id_from=333.1387.follow.user_card.click) | - |
| B站 | 毛玉仁 | [空间](https://space.bilibili.com/3546823125895398?spm_id_from=333.1387.follow.user_card.click) | - |
| B站 | 柏斯阔落 | [空间](https://space.bilibili.com/266938091?spm_id_from=333.1387.follow.user_card.click) | - |
| CSDN | v_JULY_v | [博客](https://blog.csdn.net/v_JULY_v?type=blog) | - |
| 知乎 | 锦恢 | [主页](https://www.zhihu.com/people/can-meng-zhong-de-che-xian) | - |
| 小红书 | AI有温度icefreeai | 小红书号: icefreeai | - |
| 小红书 | KI | 小红书号: 541226720 | - |

### 博客

| 文章标题/主题 | 作者 | 链接 | 备注 |
|--------------|------|------|------|
| Agent 基本概念与分类 | 锦恢 | [知乎](https://zhuanlan.zhihu.com/p/1962274523752691074?share_code=P7iG0DoioFq9&utm_psn=1974104764393490201) | Agent 小白教程 |
| 深度解析 LightRAG | 老顾聊技术 | [知乎](https://zhuanlan.zhihu.com/p/4821793882?share_code=JEwt0dQzheCt&utm_psn=1974107585884999723) | RAG 技术解析 |
| 互联网优质资源分享 | 零一猴子 | [知乎](https://www.zhihu.com/question/3946118527/answer/1919046825337398459?share_code=mllKM7MxPkU1&utm_psn=1974107684363052363) | 资源汇总 |

---

## 7. 🎮 DLC

### 算法

| 名称 | 类别 | 链接 | 备注 |
|---------|------|------|------|
| LeetCode (灵茶山艾府) | 算法博主 | [力扣主页](https://leetcode.cn/u/endlesscheng/) | - |
| NotOnlySuccess | 算法博主 | [空间](https://space.bilibili.com/3546647317448859?spm_id_from=333.1387.follow.user_card.click) | - |
| Hello算法 | 书籍 | [网站](https://www.hello-algo.com/) | 动画图解算法 |
| Deep-ML | - | [网站](https://www.deep-ml.com/) | AI 界的 LeetCode |
| codeforces-go | 代码库 | [GitHub](https://github.com/EndlessCheng/codeforces-go/tree/master) | - |
| LC-Rating工具 | 工具 | [网站](https://huxulm.github.io/lc-rating/zen) | 力扣周赛工具 |


### Others
| 名称 | 类别 | 链接 | 备注 |
|---------|------|------|------|
| build-your-own-x | 项目 | [GitHub](https://github.com/codecrafters-io/build-your-own-x) | 手搓各种技术轮子 |
