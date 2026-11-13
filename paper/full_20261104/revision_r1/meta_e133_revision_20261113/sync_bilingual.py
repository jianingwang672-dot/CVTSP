"""Mechanically retain unchanged translations and rebuild both translation maps.

Run with --before pointing to this update's pre-edit snapshot. Only changed
English units consume the explicitly authored zh_updates.txt. Unchanged units
retain their exact existing translations, including sections outside Section 5.
"""
import argparse
from pathlib import Path
import runpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=Path, required=True)
    args = parser.parse_args()
    run_dir = Path(__file__).resolve().parent
    root = run_dir.parent
    bi = root / 'bilingual'
    ns = runpy.run_path(str(bi / 'build_bilingual.py'))
    old = (args.before / 'main.tex').read_text()
    new = (root / 'main.tex').read_text()
    old_zh = ns['load_zh']([str(args.before / 'zh_part1.txt'),
                            str(args.before / 'zh_part2.txt')])
    overrides = ns['load_zh']([str(run_dir / 'zh_updates.txt')])
    old_units = {(cls, kind, text): old_zh[uid]
                 for uid, cls, kind, start, end, text in ns['units'](old)}
    units = ns['units'](new)
    zh = {}
    used = set()
    retained = 0
    for uid, cls, kind, start, end, text in units:
        key = (cls, kind, text)
        if key in old_units:
            assert uid not in overrides, f'Override on unchanged unit {uid}'
            zh[uid] = old_units[key]
            retained += 1
        else:
            assert uid in overrides, f'Missing translation {uid}: {text[:90]}'
            zh[uid] = overrides[uid]
            used.add(uid)
    assert used == set(overrides), 'Unused or stale translation override'
    zh['FIG'] = old_zh['FIG']
    start_experiments = new.index(r'\section{Computational experiments}')
    first, second = [], []
    for uid, cls, kind, start, end, text in units:
        dest = first if start < start_experiments else second
        dest.append(f'### {uid}\n{zh[uid]}\n')
    second.append(f"### FIG\n{zh['FIG']}\n")
    # These are bulk mechanical rewrites of IDs/maps, not new prose generation.
    (bi / 'zh_part1.txt').write_text('\n'.join(first))
    (bi / 'zh_part2.txt').write_text('\n'.join(second))
    (bi / 'main_bilingual.tex').write_text(ns['build'](new, zh))
    print(f'Retained {retained} translations; updated {len(used)}; total {len(units)}')


if __name__ == '__main__':
    main()
