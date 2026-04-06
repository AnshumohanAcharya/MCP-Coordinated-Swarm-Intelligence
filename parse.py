import ast, pathlib
files = [
    'experiments/failure_resilience.py',
    'experiments/communication_degradation.py',
    'experiments/scalability_study.py',
    'run_final_review_demo.py',
    'simulation/environment.py',
    'simulation/disaster_scenario.py',
]
for f in files:
    src = pathlib.Path(f).read_text()
    try:
        ast.parse(src)
        print('OK  ' + f)
    except SyntaxError as e:
        print('ERR ' + f + ': ' + str(e))