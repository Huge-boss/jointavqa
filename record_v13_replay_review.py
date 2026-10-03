"""Record visual inspection of all seven replay input sheets; not inference."""
import json
from pathlib import Path
from prepare_v11_media_review import sha

BASE=Path(__file__).resolve().parent
OUT=BASE/'offline_v13_replay_review'
NOTES=[
('R01','48.48—50.56秒为绿色夸张衣领人物；51.08秒出现米色制服女性，54.48—57.32秒为两人近身动作，59.92—60.44秒制服女性坐下。全局邻帧81.72/108.92秒及额外83/86/92秒也有该制服女性。',
 '同一人物在多个区间反复出现；帧锚点54.48秒与生成自报86秒矛盾。不能靠服装匹配决定谈论职业选择的真实语句时间。局部UNKNOWN未提供语义证据。','temporal_identity_ambiguous_requires_audio'),
('R02','锚点590.68秒及实际578.68—590.64秒输入是红色走廊内男女与第三人的互动。额外165/168/171/174/177秒诊断帧是男女在餐桌旁，男子拿着菜单状物件。',
 '生成描述自报171秒但帧地址F020指向590.68秒，场景不一致；局部UNKNOWN。已有视觉证据支持定位/场景连接错位，不能在未听证情况下判定何时解释菜肴或正确选项。','visual_context_mismatch'),
('R03','39.88—43.00秒多人在绿色房间内对峙；43.52—48.20秒可见两名男性近身抓持/冲突；48.72秒后是女性特写。所抽局部帧未显示男子落到地面。',
 '候选描述“男子摔倒”和局部描述“男子打女子”均没有在这些实际输入帧中得到直接支持。不能由稀疏帧缺失断言动作绝未发生，thud声源未听证。','action_description_not_supported_by_sampled_frames'),
('R04','2.335秒男子仍在椅旁，3.420秒已站起，4.504秒两男子站立，7.756秒一男子伸手向另一男子肩部；其后其他人起身/发言姿态。',
 '局部“站起来”有视觉支持；全局“推开”与具体动作不能仅靠这些帧确定。哪一动作发生在指定语句时仍未知，不能用图像推断语句。','visible_action_supported_utterance_alignment_unknown'),
('R05','2.971/4.055秒女子穿红外套，5.139/6.224秒在处理肩带/外衣区域；9.518/10.602秒她已穿无袖黑上衣，红衣在椅旁；11.686—14.939秒为蓝衣男子特写。',
 '可见脱外衣前后状态，但局部只复述语句并给出12秒，没描述可核验的动作联系。未听证，不能确认该句与脱衣、坐下或扶椅的时间关系，也不把离线改对当作局部事实正确证明。','event_visible_speech_alignment_unknown'),
('R06','2.292/3.375/5.542/8.833/11/14.25秒等可见短发男子；4.458/6.625/7.708/9.917/12.083/13.167秒等可见长发女子。',
 '短发/长发两类可见且对应不同人物。全局与局部描述分别绑定不同人物，需要听清语句后的说话轮次才可判断；不能仅根据可见脸或嘴姿态确定发声者。','person_binding_requires_audio'),
('R07','4.671秒为持枪男子群像，5.755—7.923秒为冒烟手提包，9.008秒为条纹西装男子与女子，10.092—12.302秒多人远景，13.386—16.639秒一男子俯身开包并有烟雾。',
 '局部确含包和人物事件，但胡须人物是否接着指定语句讲话、说的是提问还是回答均未听证，局部UNKNOWN不能解决轮次。','speaker_turn_unknown')]

def main():
    materials=json.loads((OUT/'REVIEW_INPUTS.json').read_text(encoding='utf-8'))
    records=[]
    for cid,visible,comparison,status in NOTES:
        r=next(x for x in materials if x['case_id']==cid)
        sheets=[OUT/'sheets'/f'{cid}_anchor.jpg',OUT/'sheets'/f'{cid}_local.jpg']
        if r['diagnostic_extra']:sheets.append(OUT/'sheets'/f'{cid}_claimed_time.jpg')
        records.append(dict(case_id=cid,qid=r['qid'],visible_evidence_zh=visible,comparison_and_limits_zh=comparison,
            mechanism=status,source_sha256=r['source_sha256'],crop_sha256=r['recorded_crop_sha256'],
            reviewed_sheets={str(p.relative_to(OUT)):sha(p) for p in sheets},audio_semantic_reviewed=False,
            continuous_all_frames_reviewed=False,answer_or_label_revised=False))
    result=dict(scope='All seven natural replay cases; 14 actual-input sheets plus two diagnostic generated-time sheets',
        cases=records,material_archive_sha256=sha(OUT/'review_materials.tar.gz'),
        gpu_used=False,independent_blind_review=False,semantic_audio_reviewed=0,
        limitation='Visual inspection of recorded sampled frames only; no full-motion or semantic audio verification, no causal attribution or population rate estimate.')
    (OUT/'VISUAL_REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['V13七次自然回看证据核查','7/7原片SHA与裁剪SHA核验通过，14张实际输入帧表和2张额外时间诊断表已视觉审阅；未调用模型、未改任何旧评测。','语义音频0/7，未连续审阅全帧；这不是7条答案的完整事实核验。已有结果上下文，不称独立盲审。','']
    for r in records:lines.extend([r['case_id']+' '+r['qid'],r['visible_evidence_zh'],r['comparison_and_limits_zh'],''])
    lines += ['机制判断与下一阶段',
        '裁剪请求均为12秒；三个23.976fps片段的容器视频时长约12.010008秒，属于帧边界离散化。不能把请求预算描述成编码时长严格不超过12秒。实际帧时间和SHA均记录，不修改历史输出。',
        '合法帧地址仍不保证事件定位正确。R02证明生成语义时间、帧地址及实际场景可分离；R01说明同一人物跨区间重复出现使单个人物锚点不足。此处没有证实解码器抽错帧：裁剪/索引已与记录核验一致。',
        'R03显示对实际局部事件的描述仍可失真；R04可见动作描述有支持；R05/R06/R07需要语句—人物—动作绑定，纯视觉不能补足听证。不是所有问题都能通过再看一段改善。',
        'V13形式上提供帧编号列表，模型并没有经过已验证的帧地址绑定。改善方向可以是将源时刻/帧地址与实际视觉输入显式配对，或进行候选无关的事件定位；但额外模型调用/长笔记并无收益保证。',
        '下一步先形成统一候选的设计与可实现性评估，核对两模型实际视频预处理能否支持明确地址，不改native控制、不用正确标签调地址、不盲目放宽回看门槛。若引入视觉地址覆盖层/新采样，应单列其媒体处理改变，不能把收益称纯记忆或等输入。',
        '当前证据不足以保证大幅提高，V14尚未实现/启动。下一候选必须在独立目录/分支限定最多两次新增全局前向与总token预算，冻结技术预检须检验地址到源时间一致性、native对照、两模型接口和首次成本；主方案仍须同一20%四组整体比较。',
        '不重复旧48/29材料或本次7条材料；不运行80%与组件消融，不用空闲GPU作为启动理由。语义音频无法独立核查部分保留未知，不用同模型反复解释代替。',
        '材料归档SHA256 '+result['material_archive_sha256'],
        '材料构建CPU成功轮6.844501秒，另有第一次离线构建因AV元数据字段命名不同中断并修复；第一次时间未完整记录，不宣称总研发构建仅6.844501秒。失败未运行任何推理。']
    (BASE/'V13_REPLAY_FINDINGS_ZH.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(sha(OUT/'VISUAL_REVIEW.json'))

if __name__=='__main__':main()
