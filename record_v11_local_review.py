"""Record inspected replay sheets and coverage caveats; offline only."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'offline_v11_media_review/local_coverage_audit'
NOTES={
'C04':'21.52—33.52秒主要为公交车女子，末尾有男子反打；位置线索可见，但所引语音未听证。',
'C06':'回看409.46—421.46秒为脸部有血迹男子持长物起身、离开；与诊断163.24—179.24秒捡红物并举到鼻口的事件不同。局部“站起来”有局部画面依据，但把它连接到题目红物事件缺依据。',
'C08':'5.04—17.04秒回看有司机与乘客反打；该区间不能覆盖题目10—58秒的完整后续轨迹，话语内容未听证。',
'C09':'10.36—22.36秒含电话、镜子及取下假发动作；整体情绪仍不能仅以静帧定论。',
'C10':'0—12秒前段为另一浅发女孩，10.4秒后才出现黄色花饰女孩；目标人物的局部覆盖较短。',
'C11':'25.76秒后含抱孩子、在厨房走动及接电话，相关人物可见；具体脚步声仍未独立确认。',
'C12':'0—12秒含沙发人物反打；仅对白提及物体及其首次时间不能由所查画面确认。',
'C16':'15.19—27.19秒先男子后女子，空间位置可见；“在公交车上”仍非左右位置。',
'C17':'18.83—30.83秒有男子伏在吧台、另一人物和吧台工作者；气氛/声音未听证。',
'C18':'5.52—17.52秒酒吧人物反打与后段枪械近景；Middle未定位关门事件。',
'C19':'0—12秒小卖部男孩与女子，后段男孩持棍；不能据该局部核定71秒片段全部对话顺序。烧录字幕只是视觉文字证据。',
'C20':'426.50—438.50秒是两个人物亲密接触及亲吻，未见理发工具或剪发动作；局部描述中的理发未获该输入支持。关系因果和姓名对应仍未知。',
'C21':'3.20—15.20秒公交车人物转移，有烧录Did you get it及Oh my god字幕；语音本身未听证。',
'C22':'1.68—13.68秒含吧台两人、后续插入近景；模型返回范围本身不是人物位置证据。',
'C24':'6—18秒沙发人物近景与末尾全景；无法凭静帧核定对白物体首次提及时刻。',
'C28':'1.49—13.49秒台上女子发型及男子都清晰，声音轮次仍未听证。',
'C29':'7.99—19.99秒含电话/驾驶镜头，未含已在全局22秒后可见的餐桌倒饮料镜头；回看未围绕该视觉事件。',
'C30':'0—12秒采访及身体跳动/庆祝姿态可见；you win次数仍未知。',
'C33':'5.24—17.24秒有食物上桌、光头男子品尝、多人反打；最后约2.74秒原片未纳入回看。全局音频曾包含全片，但局部文字不可声称覆盖所有品尝后时段。',
'C34':'0—12秒婚礼中蓝裙女子、儿童和不同西装男子均有画面；谁在指定话语后说话未听证。',
'C35':'0—12秒衣服与人物握手顺序清晰，包括绿色背心女子和格子衫男子；所引语句后说话人未听证。',
'C36':'1.99—13.99秒有女子坐姿至起身及后续近景；站后是否说话未听证。',
'C40':'约8.73秒倒液体、13.23秒饮用，两个动作在局部均可见；引用语句与动作的先后仍需音频。',
'C41':'与C33相同裁剪，局部可见品尝和多人；稀疏音频实际仅覆盖到原片15.985926秒，不能把请求窗口终点17.235926秒当连续音频终点。',
'C42':'0—12秒采访者与女子，后段女子近景；语速判断需要完整话语及计时，不能由出镜频率替代。',
'C44':'局部两名男子、手势和烧录glucose字幕可见；NO与原生成断言不能靠静帧裁定。',
'C45':'12.18秒伸手、13.64秒手上黄色包装、之后手势；物件和发言前后须音频/动作连续核查。',
'C46':'两名不同红/橙外衣女子均在局部，黑发男子约10.64秒有手势；语句内容及指向对象的声画绑定未知。',
'C47':'前段人物活动、11.77和13.27秒长椅六人、14.77秒裁切近景；Five未绑定所引语句时刻。',
}

def main():
    audit=json.loads((OUT/'COVERAGE_AUDIT.json').read_text(encoding='utf-8'))
    local=[r for r in audit['cases'] if 'local_sheet' in r]
    assert {r['case_id'] for r in local}==set(NOTES)
    rows=[]
    for r in local:
        p=OUT/r['local_sheet']
        rows.append(dict(case_id=r['case_id'],sheet=r['local_sheet'],sheet_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            visual_note=NOTES[r['case_id']],audio_semantics='not_reviewed',continuous_motion='not_reviewed',label_revalidated=False))
    vendor=OUT/'vendor_mm_utils.py'
    assert hashlib.sha256(vendor.read_bytes()).hexdigest()=='f5c3fb46b925c6369316d57f9e84e595a4df0c0e3710d53c47d8139052617cb3'
    doc=dict(local_contact_sheets_reviewed=29,semantic_audio_reviewed=0,records=rows,
        dense_diagnostic_review=dict(case_id='C06',intervals=[[163.24,179.24],[409.46,421.46]],fps=4,
            observation='At about163—169s the hand lifts a red item; about170—174s it is held at the nose/mouth. At409—421s a blood-marked man stands with a long object. Different events; no exact benchmark answer adjudicated.',
            evaluated_input=False),
        technical_evidence=dict(vendor_mm_utils_sha256=hashlib.sha256(vendor.read_bytes()).hexdigest(),
            function='process_audio_from_video',lines='210-238',
            finding='Recorded native snippets are concatenated then padded, and may overlap. Source-time union is not encoded sequence duration; no semantic word count inferred.'),
        limitation='Outcome-stratified qualitative development diagnostics, not an independent blinded study. No inference or score changes.')
    (OUT/'LOCAL_VISUAL_REVIEW.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Recorded29 replay-sheet reviews; semantic audio0.')

if __name__=='__main__':main()
