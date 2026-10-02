stage_roundoff_v2：独立保存数值边界修复和续跑证据。

stage_v1在Joint VideoLLaMA Agent的两个题目触发片段时长断言，两GPU退出。
原因：窗口两端保留3位小数，宽度允许12.001秒；浮点相减得到12.001000000000001，误判超限。
仅在guard增加绝对1e-12的浮点容差。实际window、ffmpeg参数、媒体、提示、模型、解码和评分均未修改。
原v5及stage_v1源码、预测和失败记录原样保留，完整旧stage归档SHA记录于repair_record.json。
所有已有成功结果逐题继承，保留origin及continuation_source路径/SHA/行号，不读取标签挑结果。
只续跑Joint VideoLLaMA Agent缺失279题，GPU0/3各独立完整模型分题。历史失败单列。
先在原2题pilot上检查原始/续跑全文、token、窗口、媒体哈希一致，通过后续跑。
模型和方法仍为开发用20%对照，不扩展全量，不把局部得分作为提升结论。
源码修改只能另起版本，不向已经运行目录混写。完成由collect_roundoff.py校验复制到本地。
