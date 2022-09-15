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
    os.chdir(f'{path_dir}')

    jupyter_files = [name for name in os.listdir(os.getcwd()) if
                     os.path.isfile(os.path.join(os.getcwd(), name)) and not name.startswith(".") and name.endswith(
                         ".ipynb")]
    for jupyter_file in jupyter_files:
        os.system(f'jupyter nbconvert {jupyter_file} --to markdown --output {jupyter_file[:-4]}')

    os.system(f'git reset')
    os.system(f'git add {path_dir}')
    os.system(f'git commit -m \"{path_dir + " - " + commit_name}\"')
    os.system(f'git push')

    os.chdir(f'..')

os.system(f'git reset')
os.system(f'git add .')
os.system(f'git commit -m \"{commit_name}\"')
os.system(f'git push')
