"""One-run local result collector. SSH credentials stay in the child environment only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
LOCAL=BASE/'v5/runs'
SSH=BASE.parents[1]/'ssh_remote.py'
REMOTE='/data/cfc/code/project/AVQA-agent/experiments/av_evidence_agent_v1/v5/runs'
COMMAND="""/root/anaconda3/envs/Joint-avqa/bin/python - <<'PY'
import json,hashlib,tarfile
from pathlib import Path
p=Path('/data/cfc/code/project/AVQA-agent/experiments/av_evidence_agent_v1/v5/runs')
s=json.loads((p/'state.json').read_text())
d=dict(state=s,progress={str(f.relative_to(p)):json.loads(f.read_text()) for f in p.glob('*/*/dev20/progress.json')})
if s['status']=='complete':
 a=p/'complete_results.tar.gz'
 try:
  with tarfile.open(a) as t:t.getmembers()
  d['archive_sha256']=hashlib.sha256(a.read_bytes()).hexdigest()
 except (OSError,EOFError,tarfile.TarError):d['packing']=True
print(json.dumps(d))
PY"""


def ssh(*args):
    return subprocess.run([sys.executable,str(SSH),*args],capture_output=True,text=True,encoding='utf-8',timeout=240,check=True).stdout


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def report():
    folder=LOCAL/'reports'
    table=json.loads((folder/'TOTAL_TABLE.json').read_text(encoding='utf-8'))
    pairs=json.loads((folder/'PAIRED_SUMMARY.json').read_text(encoding='utf-8'))
    assert len(table)==12 and all(r['complete'] and r['missing']==0 for r in table)
    names={'baseline':'直接baseline','fixed':'固定中点取证','agent':'证据记忆+最多一次回看'}
    lines=['双模型三组固定20%开发对照（v5）','',
      'JointAVBench 571题，AV-SpeakerBench 642题；两个模型均使用原冻结题目、选项和评分。',
      '每组每题独立空应用缓存。下列耗时包含取片、缓存构建、预处理、传输、生成和控制；操作系统文件缓存未清空。',
      '模型权重在每个worker内加载一次供三组共用。独立部署估算显式加回该加载成本，三组估算不能相加当作实际墙钟。','']
    for key,paired in pairs.items():
        ds,model=key.split('/')
        lines += [key,'方法 | 正确/分母 | 准确率 | 未解析 | 平均秒 | P95秒 | 平均调用 | 首次总秒(含加载) | 缓存构建秒']
        for r in table:
            if r['dataset']!=ds or r['model']!=model:continue
            lines.append(f"{names[r['method']]} | {r['correct']}/{r['planned_n']} | {r['accuracy_pct']:.4f}% | {r['unparsed']} | {r['cold_application_mean_seconds']:.3f} | {r['cold_application_p95_seconds']:.3f} | {r['mean_calls']:.3f} | {r['standalone_total_with_shared_model_load_seconds']:.2f} | {r['cache_build_seconds']:.2f}")
            lines.append(f"  严格解析次口径 {r['secondary_strict_full_denominator_pct']:.4f}%；最终生成触顶 {r['final_truncations']}；证据48token触顶 {r['evidence_token_cap_reached']}；历史错误尝试 {r['historical_error_attempts']}。")
        for comparison in ['agent_vs_baseline','fixed_vs_baseline','agent_vs_fixed']:
            p=paired[comparison];lo,hi=p['cluster_bootstrap_95pct_pp']
            lines.append(f"{comparison}: 改对{p['wrong_to_right']}，改错{p['right_to_wrong']}，净{p['net_correct']:+d}，差值{p['delta_pp']:+.4f}百分点；来源视频聚类bootstrap95%区间[{lo:.4f}, {hi:.4f}]。")
        lines.append('')
    launch=json.loads((LOCAL/'full_launch.json').read_text())
    state=json.loads((LOCAL/'state.json').read_text())
    lines += [f"两卡本轮实际队列墙钟（含加载、调度与汇总，不含pilot/版本修复）：{state['time']-launch['time']:.2f}秒。",'',
      '解释边界：这是反复开发用20%，不是独立盲测；置信区间不能消除开发选择偏差。',
      '固定组与Agent有相同2次调用上限，但实际调用数和token未必相等。固定中点是简单确定性控制，不代表优化的语义检索。',
      '两种新方法都使用同一受限字段生成；直接baseline保持原始解码。格式约束和提示也属于方法差异。',
      'Agent的时间选择仅8个格中心，证据为模型生成的有限文字；不保证音画事实正确或全局充分。',
      'Joint回看可接触原始首300秒音频以外的片段，可能增加证据覆盖；提升不能全部归因推理能力。',
      '是否值得使用须同时检查净改对、分项差异、延迟和调用成本；没有自动启动剩余80%或全量。',
      '分项/逐题/成本详见同目录JSON；完整原始输出在complete_results.tar.gz。']
    (folder/'三组对比结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    deadline=time.monotonic()+36*3600;failures=0
    while time.monotonic()<deadline:
        try:
            info=json.loads(ssh('--command',COMMAND));failures=0
            save(LOCAL/'collector_state.json',dict(status='checking',time=time.time(),remote=info))
            if info['state']['status']=='needs_diagnosis':
                save(LOCAL/'collector_state.json',dict(status='needs_diagnosis',time=time.time(),remote=info));return
            if info.get('archive_sha256'):
                ssh('--get',REMOTE+'/complete_results.tar.gz',str(LOCAL/'complete_results.tar.gz'))
                digest=hashlib.sha256((LOCAL/'complete_results.tar.gz').read_bytes()).hexdigest()
                assert digest==info['archive_sha256'],'Archive SHA mismatch'
                for name in ['state.json','full_launch.json','reports/TOTAL_TABLE.json','reports/TASK_DETAILS.json','reports/PAIRED_SUMMARY.json','reports/SETUP_TIMES.json','reports/REPORT.txt']:
                    (LOCAL/name).parent.mkdir(parents=True,exist_ok=True)
                    ssh('--get',REMOTE+'/'+name,str(LOCAL/name))
                report()
                save(LOCAL/'collector_state.json',dict(status='complete',time=time.time(),archive_sha256=digest,report='reports/三组对比结果.txt'));return
            if args.once:print(json.dumps(info,ensure_ascii=False));return
        except Exception as exc:
            failures+=1
            save(LOCAL/'collector_state.json',dict(status='collection_retry',time=time.time(),error_type=type(exc).__name__,consecutive_failures=failures))
            if args.once or failures>=12:raise
        time.sleep(180)
    save(LOCAL/'collector_state.json',dict(status='collection_timeout',time=time.time()))


if __name__=='__main__':main()
