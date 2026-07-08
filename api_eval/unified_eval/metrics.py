#!/usr/bin/env python3
import argparse
import json
import sys
from typing import Set, Dict, List, Optional


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def extract_task_number(task_id: str) -> int:
    if '/' in task_id:
        return int(task_id.split('/')[-1])
    return int(task_id)


def build_results_map(data: dict, skip_ids: Set[int]) -> Dict[int, dict]:
    results_map = {}
    for result in data.get('results', []):
        task_id = result.get('task_id', '')
        task_num = extract_task_number(task_id)
        if task_num not in skip_ids:
            results_map[task_num] = result
    return results_map


def analyze_debug_vs_code(debugger_path: str, coder_path: str, skip_ids: Set[int]) -> dict:
    debugger_data = load_json(debugger_path)
    coder_data = load_json(coder_path)
    
    debugger_map = build_results_map(debugger_data, skip_ids)
    coder_map = build_results_map(coder_data, skip_ids)
    
    common_tasks = set(debugger_map.keys()) & set(coder_map.keys())
    
    # tasks where coder passes
    coder_pass_tasks = [t for t in common_tasks if coder_map[t].get('passed', False)]
    # tasks where coder fails  
    coder_fail_tasks = [t for t in common_tasks if not coder_map[t].get('passed', False)]
    
    # debug fail | code pass
    debug_fail_code_pass_count = sum(
        1 for t in coder_pass_tasks 
        if not debugger_map[t].get('passed', False)
    )
    debug_fail_code_pass_rate = (
        debug_fail_code_pass_count / len(coder_pass_tasks) 
        if coder_pass_tasks else 0.0
    )
    
    # debug fix | code fail
    debug_fix_code_fail_count = sum(
        1 for t in coder_fail_tasks 
        if debugger_map[t].get('passed', False)
    )
    debug_fix_code_fail_rate = (
        debug_fix_code_fail_count / len(coder_fail_tasks) 
        if coder_fail_tasks else 0.0
    )
    
    return {
        'total_common_tasks': len(common_tasks),
        'coder_pass_count': len(coder_pass_tasks),
        'coder_fail_count': len(coder_fail_tasks),
        'debug_fail_code_pass': {
            'count': debug_fail_code_pass_count,
            'total': len(coder_pass_tasks),
            'rate': debug_fail_code_pass_rate,
        },
        'debug_fix_code_fail': {
            'count': debug_fix_code_fail_count,
            'total': len(coder_fail_tasks),
            'rate': debug_fix_code_fail_rate,
        },
        'skipped_ids': sorted(skip_ids) if skip_ids else [],
    }


def analyze_2stage(input_path: str, skip_ids: Set[int]) -> dict:
    ## 2-stage mode
    data = load_json(input_path)
    results_map = build_results_map(data, skip_ids)
    
    total = len(results_map)
    if total == 0:
        return {
            'total': 0,
            'python': {'passed': 0, 'total': 0, 'pass_rate': 0.0},
            'diff': {'passed': 0, 'total': 0, 'pass_rate': 0.0},
            'skipped_ids': sorted(skip_ids) if skip_ids else [],
        }
    
    python_passed = sum(1 for r in results_map.values() if r.get('python_passed', False))
    diff_passed = sum(1 for r in results_map.values() if r.get('diff_passed', False))
    
    return {
        'total': total,
        'python': {
            'passed': python_passed,
            'total': total,
            'pass_rate': python_passed / total,
        },
        'diff': {
            'passed': diff_passed,
            'total': total,
            'pass_rate': diff_passed / total,
        },
        'skipped_ids': sorted(skip_ids) if skip_ids else [],
    }


def analyze_pass_rate(input_path: str, skip_ids: Set[int]) -> dict:
    
    data = load_json(input_path)
    results_map = build_results_map(data, skip_ids)
    
    total = len(results_map)
    if total == 0:
        return {
            'passed': 0,
            'total': 0,
            'pass_rate': 0.0,
            'skipped_ids': sorted(skip_ids) if skip_ids else [],
        }
    
    passed = sum(1 for r in results_map.values() if r.get('passed', False))
    
    return {
        'passed': passed,
        'total': total,
        'pass_rate': passed / total,
        'skipped_ids': sorted(skip_ids) if skip_ids else [],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='mode', required=True)
    
    # debug-vs-code mode
    dvc_parser = subparsers.add_parser(
        'debug-vs-code',
        help='Compare debugger vs coder performance'
    )
    dvc_parser.add_argument(
        '--debugger', '-d',
        required=True,
        help='Path to debugger evaluation JSON'
    )
    dvc_parser.add_argument(
        '--coder', '-c',
        required=True,
        help='Path to coder evaluation JSON'
    )
    dvc_parser.add_argument(
        '--skip', '-s',
        nargs='*',
        type=int,
        default=[],
        help='Task ID numbers to skip/filter out'
    )
    
    # 2-stage mode
    ts_parser = subparsers.add_parser(
        '2-stage',
        help='Get python and diff pass rates from 2-stage evaluation'
    )
    ts_parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to 2-stage evaluation JSON'
    )
    ts_parser.add_argument(
        '--skip', '-s',
        nargs='*',
        type=int,
        default=[],
        help='Task ID numbers to skip/filter out'
    )
    
    # pass-rate mode
    pr_parser = subparsers.add_parser(
        'pass-rate',
        help='Get overall pass rate from evaluation file'
    )
    pr_parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to evaluation JSON'
    )
    pr_parser.add_argument(
        '--skip', '-s',
        nargs='*',
        type=int,
        default=[],
        help='Task ID numbers to skip/filter out'
    )
    
    args = parser.parse_args()
    
    skip_ids = set(args.skip) if args.skip else set()
    
    if args.mode == 'debug-vs-code':
        result = analyze_debug_vs_code(args.debugger, args.coder, skip_ids)
        print(f"Coder pass: {result['coder_pass_count']}, Coder fail: {result['coder_fail_count']}")
        dfc = result['debug_fail_code_pass']
        print(f"Debug FAIL | Code PASS: {dfc['count']}/{dfc['total']} = {dfc['rate']:.4f} ({dfc['rate']*100:.2f}%)")
        dff = result['debug_fix_code_fail']
        print(f"Debug FIX  | Code FAIL: {dff['count']}/{dff['total']} = {dff['rate']:.4f} ({dff['rate']*100:.2f}%)")
        
    elif args.mode == '2-stage':
        result = analyze_2stage(args.input, skip_ids)
        py = result['python']
        print(f"Python: {py['passed']}/{py['total']} = {py['pass_rate']:.4f} ({py['pass_rate']*100:.2f}%)")
        diff = result['diff']
        print(f"Diff:   {diff['passed']}/{diff['total']} = {diff['pass_rate']:.4f} ({diff['pass_rate']*100:.2f}%)")
        
    elif args.mode == 'pass-rate':
        result = analyze_pass_rate(args.input, skip_ids)
        print(f"Passed: {result['passed']}/{result['total']} = {result['pass_rate']:.4f} ({result['pass_rate']*100:.2f}%)")
    
    if skip_ids:
        print(f"\nSkipped task IDs: {sorted(skip_ids)}")
    
    print()


if __name__ == '__main__':
    main()

