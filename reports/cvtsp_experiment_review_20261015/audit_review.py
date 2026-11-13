"""Check the delivered report against the captured numerical evidence."""
import csv
import json
import math
import re
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

OUT=Path(__file__).resolve().parent
STEM='CVTSP_实验完整复盘_500epoch至LKH'
data=json.loads((OUT/'evidence_snapshot.json').read_text())
md=(OUT/f'{STEM}.md').read_text()
checks=[]


def check(name,passed,detail=None):
    checks.append({'requirement':name,'passed':bool(passed),'detail':detail})
    if not passed:
        raise AssertionError((name,detail))


check('23 numbered main sections plus 5 appendices',
      re.findall(r'^## (\d+)\.',md,re.M)==[str(i) for i in range(1,24)]
      and re.findall(r'^## 附录([A-Z])\.',md,re.M)==list('ABCDE'))
check('all requested experiment families included',all(word in md for word in
      ['S500','R200','IL2','LR150','swap-only','2-opt-only','near-miss','QIL-B','LKH-pilot','LKH100']))
check('no malformed formula/control characters',not any(ord(c)<32 and c not in '\n\t' for c in md)
      and r'\frac{V_E-V_T}' in md and r'\frac{V_T-V_E}' in md)
check('unfinished experiments identified',all(s in md for s in
      ['实际完成6轮','2256/5000','尚未实施','新test500正式评测']))
check('training completion comes from saved metric rows',
      data['run_latest']['S500']['last_completed_epoch']==500
      and data['run_latest']['R200']['last_completed_epoch']==200
      and data['run_latest']['QIL-B']['last_completed_epoch']==6)
check('completed controlled experiments audited',all(a['passed'] for a in data['audits'].values()))
check('latest LKH run began from scratch',data['lkh_initialization']['fresh_initialization']
      and data['lkh_initialization']['replay_size']==0 and data['lkh_initialization']['start_epoch']==1)
for group,epoch in [('S500',100),('R200',80),('LKH100',46)]:
    rr=[r for r in data['selection'] if r['run']==group]
    best=min(rr,key=lambda r:r['mean_makespan'])
    check('selection minimum '+group,best['epoch']==epoch)

for row in data['external_results']:
    if row.get('baseline_mean') is not None and row.get('gap_percent') is not None:
        calc=100*(row['model_mean']/row['baseline_mean']-1)
        check('external gap '+str((row['run'],row['epoch'],row['dataset'],row['J'])),
              math.isclose(calc,row['gap_percent'],abs_tol=1e-9))
    if all(isinstance(row.get(k),int) for k in ['wins','ties','losses']):
        check('paired count '+str((row['run'],row['epoch'],row['dataset'],row['J'])),
              row['wins']+row['ties']+row['losses']==row['n'])

groups={}
for row in data['external_results']:
    groups.setdefault((row['run'],row['epoch'],row['dataset']),[]).append(row)
for key,rr in groups.items():
    allrow=next((r for r in rr if r['J']=='all'),None)
    parts=[r for r in rr if r['J']!='all']
    if allrow and parts:
        count=sum(r['n'] for r in parts)
        value=sum(r['n']*r['model_mean'] for r in parts)/count
        check('weighted mean per J '+str(key),count==allrow['n']
              and math.isclose(value,allrow['model_mean'],abs_tol=1e-8))

steps={(r['run'],r['epoch']):r for r in data['optimizer_steps']}
check('equal-sample IL steps verified',steps[('S500',100)]['instances']==steps[('LKH100',50)]['instances']==1000000
      and steps[('S500',100)]['il_steps']==40510 and steps[('LKH100',50)]['il_steps']==21165)
check('LKH E20 full-POMO stage inference remains qualified',
      next(r for r in data['lkh_stage_transitions'] if r['epoch']==20 and r['mode']=='full_pomo_greedy')['ci95'][0]<0
      and '不能据此认定IL整体有害' in md)
for path in re.findall(r'\]\(<([^>]+)>\)',md):
    check('source link exists '+Path(path).name,Path(path).exists())

with zipfile.ZipFile(OUT/f'{STEM}.docx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            ET.fromstring(z.read(name))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    root=ET.fromstring(z.read('word/document.xml'))
    text=''.join(root.itertext())
    check('Word content and numeric tables survived export',all(s in text for s in
          ['CVTSP','333.459454','334.438561','17.904274','待补充','19.4']))
    check('Word embeds both figures',len([n for n in z.namelist() if n.startswith('word/media/')])==2)
    check('Word contains comparison tables',len(root.findall('.//w:tbl',ns))>=38)
html=(OUT/f'{STEM}.html').read_text()
check('HTML self-contained figures and no external scripts',html.count('data:image/')==2
      and not re.search(r'<script[^>]+src=["\'](?:https?:)?//',html))

pdf=OUT/f'{STEM}.pdf'
if pdf.exists():
    info=subprocess.run(['pdfinfo',str(pdf)],check=True,stdout=subprocess.PIPE,universal_newlines=True).stdout
    (OUT/'pdfinfo.txt').write_text(info)
    subprocess.run(['pdftotext','-layout',str(pdf),str(OUT/'pdf_extracted_text.txt')],check=True)
    pdftext=(OUT/'pdf_extracted_text.txt').read_text()
    check('PDF rendered core conclusions and numbers',all(s in pdftext for s in
          ['333.459454','334.438561','17.904274','附录']))
    pages=re.search(r'Pages:\s+(\d+)',info)
    check('PDF has multiple readable pages',pages and int(pages[1])>=20,info)

(OUT/'completion_audit.json').write_text(json.dumps({'passed':True,'snapshot':data['snapshot'],
      'checks_count':len(checks),'checks':checks},ensure_ascii=False,indent=2))
print(json.dumps({'passed':True,'checks':len(checks),'characters':len(md),
                  'source_files':len(json.loads((OUT/'source_manifest.json').read_text())['sources'])},ensure_ascii=False))
