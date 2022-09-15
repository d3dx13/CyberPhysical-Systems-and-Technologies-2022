import os
import sys

commit_name = ' '.join(sys.argv[1:])
if len(commit_name) == 0:
    commit_name = 'auto upload'

path_dirs = [name for name in os.listdir(os.getcwd()) if
             os.path.isdir(os.path.join(os.getcwd(), name)) and not name.startswith(".")]

os.system(f'git config credential.helper store')
os.system(f'git pull')

for path_dir in path_dirs:
    os.system(f'cd {path_dir}')

    os.system(f'git reset')
    os.system(f'git add {path_dir}')
    os.system(f'git commit -m \"{path_dir + " - " + commit_name}\"')
    os.system(f'git push')

    os.system(f'cd ..')

os.system(f'git reset')
os.system(f'git add .')
os.system(f'git commit -m \"{commit_name}\"')
os.system(f'git push')
