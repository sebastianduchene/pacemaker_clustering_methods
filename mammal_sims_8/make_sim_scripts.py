import re

lines = open('run_mammals_8.py', 'r').read()

for i in range(100):
    lines_new = re.sub('REP', str(i), lines)
    open('rep_'+str(i)+'.py', 'w').write(lines_new)

