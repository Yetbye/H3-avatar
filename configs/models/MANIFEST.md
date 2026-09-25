# 模型资产清单

> 更新日期：2026-09-25

## Wan2.2-TI2V-5B

- 来源：ModelScope 官方仓库 `Wan-AI/Wan2.2-TI2V-5B`
- 下载 revision：`master`
- 本地路径：`checkpoints/Wan2.2-TI2V-5B`
- 文件数：22（不含本地生成的校验清单）
- 目录字节数：34,203,126,279（包含本地 `SHA256SUMS.local` 时的统计）
- 完整 SHA-256 清单：`checkpoints/Wan2.2-TI2V-5B/SHA256SUMS.local`
- 状态：下载完成；safetensors 索引中的 3 个分片均存在；尚未执行模型加载或推理。

核心文件 SHA-256：

```text
720b06c4ade5e87c1246bba8ac95b664c638749cd9b102cf84d823bb44c026a1  diffusion_pytorch_model-00001-of-00003.safetensors
09ec5ef720d8396f6cfa51fbdcbdb2327e37722afd6e89fd38f1e7e5e782c283  diffusion_pytorch_model-00002-of-00003.safetensors
6306f7894c345de9093ad588771c2abfaeb668a81f7a6d9a918bd26ba3568e49  diffusion_pytorch_model-00003-of-00003.safetensors
7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d  models_t5_umt5-xxl-enc-bf16.pth
20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36  Wan2.2_VAE.pth
```

注意：ModelScope 下载入口只记录了浮动的 `master`，因此以上 SHA-256 才是本地资产的不可变标识。

## TalkVerse S2V 5B

- 原始来源：Hugging Face `snap-research/talkverse-s2v-5b`
- 下载源：`https://hf-mirror.com`（公开仓库匿名下载，未发送 token）
- 固定 revision：`3a58ee5632c0cf65808bdcb5be2a35e309462e4b`
- 本地路径：`checkpoints/talkverse-s2v-5b/talkverse_s2v_5b.safetensors`
- 大小：2,095,794,778 字节
- SHA-256：`07dd12fdd2044e6f9594009a8f24db4b4fe0a36b1c2cd3d30c818c31c7713923`
- 验证：safetensors 头可读取，共 640 个 tensor；尚未与 Wan2.2 完成联合加载。

## Wav2Vec2 XLSR-53 English

- 原始来源：Hugging Face `jonatasgrosman/wav2vec2-large-xlsr-53-english`
- 下载源：`https://hf-mirror.com`（公开仓库匿名下载，未发送 token）
- 固定 revision：`569a6236e92bd5f7652a0420bfe9bb94c5664080`
- 本地路径：`checkpoints/Wan2.2-TI2V-5B/wav2vec2-large-xlsr-53-english`
- 下载范围：`model.safetensors` 及 Processor 所需的 5 个 JSON 配置；未下载重复的 PyTorch/Flax 权重和可选语言模型。
- 模型 SHA-256：`6144f8464c6aaa220dd57c5a2ad4039b5710dcf8ee6e67057675f76597c19875`
- 验证：`Wav2Vec2Processor` 与 `Wav2Vec2ForCTC` 本地加载成功，315,472,545 参数，词表大小 33。
