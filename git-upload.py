import os
import sys
import subprocess

OLD1 = r"\documentclass[11pt]{article}"
NEW1 = r"""
\documentclass[11pt]{article}
    \usepackage[utf8]{inputenc}
    \usepackage[T1,T2A]{fontenc}
    \usepackage[russian,english]{babel}
"""
REPLACE = [(OLD1, NEW1)]

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
        jupyter_name = jupyter_file[:-6]
        subprocess.call((f'jupyter nbconvert {jupyter_file} --to markdown --output {jupyter_name}'))
        subprocess.call((f'jupyter nbconvert {jupyter_file} --to latex --output {jupyter_name}.tex'))

        with open(f'{jupyter_name}.tex', 'r', encoding="utf-8") as file:
            filedata = file.read()
        filedata = filedata.replace(OLD1, NEW1)
        with open(f'{jupyter_name}.tex', 'w', encoding="utf-8") as file:
            file.write(filedata)

        subprocess.call((f'pdflatex -interaction=batchmode {jupyter_name}.tex -output-format pdf'))

        os.remove(f'{jupyter_name}.tex')
        os.remove(f'{jupyter_name}.log')
        os.remove(f'{jupyter_name}.aux')
        os.remove(f'{jupyter_name}.out')

        # os.system(f'jupyter nbconvert {jupyter_file} --to pdf --output \"{f"{jupyter_file[:-6]} - Отчёт Жидков А.А. R4136с "}\"')
        # os.system(f'pandoc -V lang=russian -o {jupyter_file[:-6]}.pdf -f markdown --pdf-engine=pdflatex {jupyter_file[:-6]}.md')
        # pandoc MANUAL.txt --pdf-engine=xelatex -o example13.pdf  --pdf-engine=pdflatex

    os.system(f'git reset')
    os.system(f'git add .')
    os.system(f'git commit -m \"{path_dir + " - " + commit_name}\"')
    os.system(f'git push')

    os.chdir(f'..')

os.system(f'git reset')
os.system(f'git add .')
os.system(f'git commit -m \"{commit_name}\"')
os.system(f'git push')
