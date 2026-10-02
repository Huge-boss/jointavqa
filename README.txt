当前新实验：v3/，三组分别为重新实测baseline、固定预算中点取证、题内证据记忆加最多一次按需回看。
独立分支experiment/av-memory-agent-v3-20pct；配置与计时口径见v3/README.txt和protocol.json，状态以v3/inspect.sh为准。
v2四轮版本于2026-10-02 08:26:50 UTC因用户选择更轻量协议而停止；AV-Speaker已完成Omni206题、Video247题，未据得分决定停止。
v2_stop_record_for_v3.json记录进程和原始输出SHA；完整部分证据已复制本地v2_partial_evidence_before_v3.tar.gz，不混入v3。
根目录v1和v2均为保留的历史版本，禁止重启其launch.sh/run_campaign.py；v2技术记录仍见v2/REPAIR.txt。

免训练音画证据 Agent：首轮固定20%对照（2026-10-02）

目标与范围
只使用Qwen2.5-Omni-7B和VideoLLaMA2.1-7B-AV，所有模型权重冻结，不训练、不下载额外模型。
第一轮验证主动选择音画片段是否有效，不承诺大幅提升，不按分数重新抽题或改参数。
代码、结果和缓存位于独立experiment目录，旧评测源码、输出、标签与环境保持原样。

文献依据（2026-10-02查阅一手论文）
1. VideoAgent: Long-form Video Understanding with Large Language Model as Agent
   https://arxiv.org/abs/2403.10517
   借鉴其交互式查证思路；本版不声称复现它的模型组合、检索器或论文成绩。
2. VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos
   https://arxiv.org/html/2405.19209v2
   借鉴问题相关区域的由粗到细检查；本版没有实现其完整聚类树。
3. Daily-Omni v2, Section3.3
   https://arxiv.org/html/2505.17862v2
   借鉴定位关键事件、回取同步音画证据再汇总。该文组合多个模型，本版只用同一个基础模型分别执行观察、选择和作答，避免把新增大模型能力混进改进。

实验流程
原始直接作答：复用已有完整原始输出，在本次完全相同题号上用冻结评分器重算。
固定流程：全局观察 → 1/3处12秒片段 → 2/3处12秒片段 → 原全局输入加观察记录作答。
Agent：全局观察提出SEEK时间 → 查看片段并提出第二个SEEK → 再查看片段 → 原全局输入加记录核对选项后作答。
两种新方法各4次逻辑调用，共享同一全局观察，实际每题计算7次；局部窗口时长一致。
第二次Agent选择真正读取第一次工具结果，不是预先固定两段。非法工具参数采用固定位置回退并记录。
模型输出只作为文本和受限数字，不作为可执行指令。片段不能超出官方发布片段，不能读取未来视频。
全程记录原始输出、时间窗口、媒体哈希、完整提示、帧数、音频量、token、耗时、截断和回退。
旧Joint视频本身可能带烧录字幕，不额外输入字幕文件、ASR或标注描述。

子集与评分
JointAVBench571/2853题，15类；AV-SpeakerBench642/3212题，12类。
每类按20%最大余数分配，固定salt/seed42和题号SHA256排序，两个模型和两条新流程使用同一题目与选项。
抽样脚本不打开标签和旧预测。manifests/*_split.json保留选中及其余题号。
这20%用于开发迭代；同视频问题可能跨分区，加上旧全量和诊断已经看过，因此不能把余下80%宣传为完全未见测试集。
Joint统一使用先前冻结的parser_v2；历史原分仍保留。AV-Speaker同时报告上游first-letter和事前strict解析。
未知/未解析/运行错误不得猜答案或删题。evaluate.py独立读标签，不被worker导入。
只在完整配对子集上报告增减题、类别准确率、来源视频聚类bootstrap区间。4次调用是次数预算相同，不等于FLOPs或实际token严格相同。
首轮不自动跑80%或全量，也不据中途分数停止或改提示。结果不好同样保留并报告。

隔离与恢复
compat/是原有两模型适配器的快照，只修正相对目录定位；外部官方VideoLLaMA实现固定SHA，旧文件不改。
新worker开始前核对源码、输入、协议和外部依赖SHA。变化必须另开版本和输出目录。
仅GPU0 Omni与GPU3 Video并行，每卡一个完整模型。先完成两个数据集各最短/最长题pilot，门槛只检查执行、输入一致性和工具可用性，不看正确率。
同一版本OOM至多三次，其余失败停下诊断；旧调度器不启动。断点只跳过同配置已成功题，不混入其他版本。
GPU0和3启动前必须空闲；不会终止其他任务腾卡。

运行（服务器已有模型/数据/环境）
1. python prepare.py
2. python freeze_remote.py（探测输入时长、冻结pilot/依赖；只在未启动前）
3. python test_contracts.py
4. /root/anaconda3/envs/Joint-avqa/bin/python run_campaign.py
5. runs/reports/下自动产生总表JSON、分项JSON、逐题对照和REPORT.txt；完成生成complete_results.tar.gz。
实际运行遵守已冻结manifests/code_freeze.json。Git提交/tag加源码SHA共同定位版本，数据/模型/密码/token不进Git。

后续如何判断
Agent高于直接作答但与固定流程接近：主要支持多看/分段的作用，不能声称主动策略有效。
Agent稳定高于固定流程：再检查关键类别、失败回退、资源开销，设计去掉第二次查证/去掉时间信息等消融。
若Agent下降：保留负结果，先从选片错误、错误观察传播、说话人错绑分析，不用换题/改解析器追分。
当前是基于现有思路的可执行起点；创新性和显著提升都需要结果与对照支持。
