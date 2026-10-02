"""Record offline visual inspection; never imported by inference or scoring."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'offline_v11_media_review'
# These are qualitative observations of the saved source-frame sheets, not labels.
OBS = {
1: '输入帧可见沙发上的两人及部分站立人物；未据这些静帧确定关门事件及声源位置。',
2: '0—2.72秒有手、笔和纸，3.28秒后有驾驶者及乘客。笔记集中在手和点击，未保留多事件次序；对话先后未知。',
3: '0—5.04秒为车窗外夜间城市。视觉概括有支持，音乐与整体情绪分类未核实。',
4: '输入含公交车上红色上衣女子与反打男子；局部描述“在公交车上”没有回答左右位置，所引语句的说话人未核实。',
5: '窗帘旁浅色短发人物已进入输入；所引“Oh, no”与人物的绑定未核实。',
6: '530.88秒片段仅32帧，最大间隔约17.12秒。171.24秒附近手已靠近面部，但红色物体及紧接动作需连续片段核实，不能直接判定嗅闻。',
7: '所查帧主要有一名清晰可见男子拉窗帘或经过门口；笔记“三个人走过门口”未得到这些帧支持，不能排除帧间人物。',
8: '笔记把10—58秒概括为女子在车内；实际输入32.28秒起可见楼梯、站台、列车和闸机，概括与输入后半段明显冲突。对话仍未听证。',
9: '输入有男子在浴室镜前打电话，并在13.68—15.12秒附近取下头上物件；笔记较泛，情绪与音乐未核实。',
10: '黄色花饰女孩及其他人物均在输入中；跨镜头人物绑定和情绪判断仍需原片及音频。',
11: '戴白头巾并抱孩子的女子、走动与电话画面均进入输入；局部“鞋子发出声音”没有定位具体人物。',
12: '沙发人物、发带、胡须等可见；要求的仅对话提及物体及首次时间需要音频，不能采用模型给出的精确时间当证据。',
13: '与C01同片。笔记“很可能是客厅门”属于推测，所查画面未能定位具体关门声源。',
14: '红发红上衣女子及另一男子清晰进入输入；情绪类别和对白语气未核实。',
15: '笔记称画面只有黑底白色时钟；实际32帧含人脸、田野、森林及人物动作，存在直接视觉矛盾，并非全部相关画面都未输入。',
16: '公交车人物及衣服进入输入；局部“红衣女子在公交车上”仍缺左右空间位置。模型关于声音性别的断言尚未核实。',
17: '多段酒吧人物和吧台可见；笔记所称镜面反射关系在所查帧中不能确认，整体气氛依赖额外声画证据。',
18: '酒吧女子、牛仔帽男子及吧台已输入；局部“Middle”是位置答案式输出，没有提供关门事件定位证据。',
19: '71.4秒输入跨多个场景，11.52秒附近男孩持棍，69.08秒出现另一场景的烧录字幕；0—12秒局部笔记却给出全题事件顺序。该局部范围不能单独证实后段事件次序；字幕不等于已听证语音。',
20: '768.84秒32帧，间隔约24.8秒；446.44秒后仍有人物互动。首300秒音频预算及稀疏帧可能限制长片因果证据。笔记“理发”未由所查帧建立，关系原因仍未知。',
21: '男子持绿色苹果及女子均进入输入，部分烧录字幕可见；局部新增手机等物件未得到所查帧支持，对话顺序须听证。',
22: '女子及吧台旁男子可见；局部返回时间区间而非人物位置，未提供所需空间关系。“看手机”在所查帧中未确认。',
23: '灰发卷发蓄须男子清晰进入输入；情绪与声音语气未核实。',
24: '与C12同片；局部提及胡须、夹克和时间范围，没有建立题目要求的对话物体及首次出现时间。',
25: '持麦克风两人及麦克风转向过程可见；笔记没有可审计的college逐次出现时间或计数，音频语义未核实。',
26: '4.67秒画面同时包含男子和右侧躺着的女子，5.88秒也有女子，7.05秒附近女子起身；“醒前只有一人可见”的概括有明显人数遗漏风险。醒来边界和叠加图片计数口径需连续片段确认。',
27: '输入有最初同行两人、后出现的黑裙女子及拥抱过程；“两人走路”不足以保存多人物事件时间线，costume party出现时刻未知。',
28: '台上卷/波浪发女子与短发男子均清晰可见；外观证据已进入，引用语句后的说话人绑定仍需音频。',
29: '汽车、厨房电话及餐桌倒饮料相关画面均进入输入；谁在倒果汁期间说话不能由画面主体直接推出。',
30: '街头采访及跳动相关姿态可见；“you win”次数和跳跃时窗必须听证，不能采用重复笔记作为第二份事实证据。',
31: '白色上衣和灰色T恤两名男子可见；笔记的第一/第二说话人未与服装稳定对应，音高无法由静帧判断。',
32: '采访画面有三名主要人物；可见人数与实际说话人数是不同量，后者及指定话语前的边界未核实。',
33: '厨师场景左侧三人之外，右侧还有部分可见的第四名光头男子，2.33、3.50及16.43秒等帧均有证据。笔记“三人可见”遗漏部分可见人物；实际说话人数仍未知。',
34: '蓝裙女子、黑西装男子和另一男子均进入输入；全局与局部笔记对下一说话人的服装绑定不一致，须音频核定，不能按标签选择某条笔记。',
35: '胡须男子、绿色背心女子和格子衫男子均进入输入；“绿色上衣”缺身份锚点，所引语句后人物绑定仍未知。',
36: '女子持物、起身及后续面部画面已输入；全局与局部关于是否说话的文字结论有分歧，站起后的声音必须听证。',
37: '与C28同片；两种发型已进入8帧输入，但泛化人物笔记没有保留发型与轮次绑定。',
38: '与C25同片，输入清晰包含两个人物和麦克风；观察字段UNKNOWN没有形成计数和事件时间证据。',
39: '8.05秒附近女子举起钥匙，其他输入帧有持食物及伸手动作；笔记缺少钥匙事件时间锚点，拿钥匙后是否说话未知。',
40: '8.72秒倒液体、13.72秒饮用的两个候选动作均进入输入；局部“倒饮料”有动作画面支持，但与“What is your name”先后关系未知。',
41: '与C33同片，8帧中也含部分可见第四人及品尝动作；局部“Three”缺少所数对象和时间界限，不能当人数事实。',
42: '采访者与女子在输入中，后段女子近景较多；最高语速需要声音与时间量，不能从面部静帧或出镜时长推断。',
43: '输入含绿衣女子、蓝色上衣男孩、粉衣女子及戴红帽男子；笔记的少数人物概括未覆盖全部候选，最低音量仍需音频。',
44: '8帧均有两名男子，眼镜男子多次手势，早段烧录字幕可见；局部NO与全局文字断言不同，但说话与“not so much”边界不能仅靠静帧核定。',
45: '眼镜男子及另一西装男子进入输入，12.89秒附近有黄色包装物；物件确切名称及拿到前是否说话仍未知，局部NO未给事件证据。',
46: '输入有至少两名不同红/橙色外衣女子，10.26秒男子有指向动作；全局与局部给出的引语不一致，衣服 referent 和动作前语句须原音核实。',
47: '1.04秒可见五人，11.67及13.79秒长椅上可见六人；人数随场景变化。局部Five没有定位“You know you could sue”的时刻，不能仅据某一帧人数裁定。',
48: '1.87及3.10秒附近红橙/蓝上衣男子与标牌、条纹球衣人物均进入输入；笔记“男子指牌说话”未提供所问语句内容。',
}

def main():
    cases = json.loads((OUT/'BLIND_REVIEW.json').read_text(encoding='utf-8'))
    assert len(cases) == len(OBS) == 48
    reviewed=[]
    for c in cases:
        n=int(c['case_id'][1:])
        reviewed.append({
            'case_id':c['case_id'], 'qid':c['input']['qid'],
            'dataset':c['dataset'], 'model':c['model'],
            'source_sha256':c['source_sha256'],
            'inspected_artifact':c['sheets'][0],
            'inspected_artifact_sha256':hashlib.sha256((OUT/c['sheets'][0]).read_bytes()).hexdigest(),
            'visual_review':OBS[n],
            'scope':'All recorded global source frames viewed as JPEG contact sheet before model resizing; comparison with generated observation text.',
            'semantic_audio_review':'not_performed',
            'continuous_motion_review':'not_performed',
            'ground_truth_revalidated':False,
            'exact_answer_adjudicated':False,
        })
    doc={
        'review_type':'qualitative_source_frame_and_note_comparison',
        'reviewer':'AI assistant visual inspection; not independent human adjudication',
        'sample_selection':'48 deterministic outcome-stratified development cases; not representative and not fully blinded. Reviewer materials omit labels, but selection and prior context use offline outcomes.',
        'source_archive_sha256':'1cf3072857114ffa711069e3edb55aff4e6cb198cd00135ae2a10307e07f3627',
        'materials_archive_sha256':'7e5ec4feda504ee4eb243190032837681153a0203ff3c4e836909da14d4a4295',
        'completed_global_contact_sheet_reviews':48,
        'completed_semantic_audio_reviews':0,
        'all_factual_reviews_complete':False,
        'cases':reviewed,
    }
    (OUT/'VISUAL_REVIEW.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'recorded':48,'audio_reviewed':0,'labels_changed':False}))

if __name__ == '__main__':
    main()
