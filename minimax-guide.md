# minimax\-guide

> TaoMate\-H3是最大的竞品但还不成熟，所以细节可以参考它，很多工作主要参考他改造

# dataset and env

TalkVerse数据集，最佳数据集：[TalkVerse: Democratizing Minute\-Long Audio\-Driven Video Generation](https://zhenzhiwang.github.io/talkverse/)

### env！！！！！！

先看minimax的cuda版本，不一致先conda install cuda toolkit \-\-label  对齐，第二步conda安装对应cuda的pytorch，都要问AI强制固定版本，不然智障的conda会给你安装他觉得最合适的
 对于flashattn、sageattn 这类与底层硬件的协同的包，一定要有 \-\-no abloation，如果显示没有这个版本就是没上架到conda和pip渠道，不需要更新pip，去git clone 这些仓库 ，checkout到那个版本，后面问AI怎么变成wheel加入环境。
**所有一定要走国内镜像，不然慢四，模型走modelscope能断点下载**

# train

> 训练代码和模型架构，我会同步探索，但是隔离,各做各的。有问题多交流\!

### 需要先去阅读liveact这篇论文，了解数字人模型结构

需要注意的是minimax是原生音画同步，和wan 音频驱动画面完全不一样，需要改接口

### 只有微调minimax才能训练，使用lora只微调自注意力层，参数规模为0\.5%到6\.5%。

代码在H3\-world

### AR\+distillation

把一个离线视频生成模型后训练为实时交互，需要双向后训练，加causal mask 蒸馏为AR模型，分布初始化，步数蒸馏。
**你git clone minwm 和solarwm\-H3 的框架让AI给你讲**
 对于**原生音画同步如何交互**我也不知道，最简单比如提示词改造，或者训一个端到端的统一transformer。
**原生音画同步如何交互**你让ai调研一下，之后讨论，目前我没有看到相关论文

# dev \& infra

部署教程：[MiniMax H3本地部署教程 \- 小红书](https://www.xiaohongshu.com/explore/6a97e16f000000000b00edf4?xsec_token=ABMVVZnsy1acxYOwRjr-GbJ9nkMqW-XkjHNE4zU-4ExEI=&xsec_source=pc_search&source=web_explore_feed)
 先看看TaoMate\-H3和 **nvidia Sol\-H3**的能不能finetune，不行的话调研一下再确定基模，比如量化版
 记住只lora微调自注意力。
 对于fal\-H3 max不知道开源了没

infra主要看；
[🎬vLLM\-Omni \+ FastH3：视频生成快过播放 \- 小红书](https://www.xiaohongshu.com/explore/6a97bd440000000027017d40?xsec_token=ABMVVZnsy1acxYOwRjr-GbJz9TrE_eCrn61KnvaBQULn0=&xsec_source=pc_search&source=web_explore_feed)
[VDN\-H3开源，视频生成快过播放 \- 小红书](https://www.xiaohongshu.com/explore/6a9a44d1000000000b001d1a?xsec_token=ABH6WOdb3rxCe-qj38_2GtjjiczWvPptcir4FhePIn5HU=&xsec_source=pc_search&source=web_explore_feed)
 和模型训练是解耦的

# system

现在项目的实时视频系统里是基于live什么改造的，就是上次部署到你的电脑上的，但是没有适配现在的模型，你需要修改测试，可以部署一个flashhead1.3B大小的测试

### Agent 驱动系统

我暑假已经搭好了基础版本，后续你们看功能优化，这块目前不是重心！！！

# paper\-github\-转化

* [ ] HCI 定义好一个未来年轻一代的虚拟交互范式
* [ ] Video gen 对打liveact等
