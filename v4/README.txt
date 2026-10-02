免训练音画证据记忆：三组固定20%对照（v4）

v3的VideoLLaMA pilot只输出ANSWER，五字段记忆格式检查失败，未启动全量。v4仅简化为EVIDENCE、SEEK、ANSWER三字段并把答案放最后；无标签诊断，无得分调参。

仅Qwen2.5-Omni-7B/GPU0和VideoLLaMA2.1-7B-AV/GPU3。Joint571题，AV-Speaker642题，沿用v2冻结题目和选项。
旧v1/v2/v3和原始baseline全部保留；不覆盖、不因得分重抽、不训练。继续使用新分支experiment/av-memory-agent-v3-20pct，实际源码和结果在v4。

三组
baseline：重新运行原生直接回答；保留原始prompt/256新token上限，获取本轮可比耗时。
fixed：全片候选答案+短证据记忆，再看中点12秒，同一次回看直接给最终答案。固定2次调用。
agent：相同首轮，只有缺失/冲突证据或格式错误时回看一次；候选答案和证据作为题内工作记忆。最多2次调用。
固定组是固定预算、均匀中点的证据选择对照，不宣称语义检索。Agent以模型提出的SEEK决定位置，无效请求使用中点并记录。
两组首轮128token、最终16token。初次记忆截断必触发一次回看。字段歧义不得直接早停。

计时与缓存
每组每题都从空的应用层媒体缓存开始，最多2个CPU预处理结果；计时包含清空、片段裁剪、缓存构建、processor、数据传输、生成和控制逻辑。
模型权重加载一次供三组使用，单独记录加载耗时；报告独立部署总耗时时显式加回相同加载成本，不能将三组的独立总时长相加当作实际墙钟时间。
操作系统文件缓存未清空，不能称硬件冷启动。warm-cache只在pilot单独测，不替代主结果。无跨题答案记忆，无KV缓存复用。
使用CPU预处理缓存提高重复检查效率；本轮每题不同窗口可能均无命中，不承诺缓存本身能提升冷启动速度。
保留无损x264/PCM16裁剪以沿用已核验的音画边界；没有声称零裁剪开销或已实现全部内存取片。

先做pilot-only，通过不看正确率的完整性检查后启动正式队列；每条lane依次AV-Speaker、Joint。所有失败记录保留。
输出v4/runs下三组原始输出、控制决策、逐调用成本、配置和哈希。
汇总TOTAL_TABLE.json/TASK_DETAILS.json/PAIRED_SUMMARY.json/SETUP_TIMES.json；包括准确率、改对/改错、平均/P50/P95延迟、调用次数、首次应用缓存总耗时。
开发集不是独立盲测；实验不足以承诺大幅提升，也不以论文总分作为调参目标。

设计参考（仅借鉴，不复现其模型组合或宣称同等指标）
VideoAgent memory: https://arxiv.org/html/2403.11481v2
VideoTree: https://arxiv.org/html/2405.19209v2
Video-RAG: https://arxiv.org/html/2411.13093v4
Daily-Omni diagnostic alignment: https://arxiv.org/html/2505.17862v2
ReMem: https://github.com/jinlab-imvr/ReMem
我们的工作假设：题内时间证据+有上限的回看减少固定多轮开销；实际收益需三组同题验证。
