import os
import sys
import subprocess

OLD = [
    r"\documentclass[11pt]{article}",

]
NEW = [
    r"""\documentclass[12pt]{report}
    \usepackage[utf8]{inputenc}
    \usepackage[T1,T2A]{fontenc}
    \usepackage[russian]{babel}
""",

]


def make_title(filetext: str, header):
    subject = "По предмету \"Киберфизические системы и технологии\""
    author = "работу выполнил: Жидков Артемий Андреевич"
    group = "группа: R4136с"
    teacher = "преподаватель: Афанасьев Максим Яковлевич"
    date = "дата: сентябрь 2022"

    title_find = r"\maketitle"
    title_text = r"\title{" + f"{header} \\\\ {subject}" + r"}" + "\n"
    title_text = title_text + r"    \author{" + f"{author} \\\\ {group}" + r"}" + "\n"
    title_text = title_text + r"    \date{" + f"{teacher} \\\\ {date}" + r"}" + "\n"
    title_text = title_text + r"    \maketitle" + "\n"

    print(title_text)
    filetext = filetext.replace(title_find, title_text)
    return filetext


commit_name = ' '.join(sys.argv[1:])
if len(commit_name) == 0:
    commit_name = 'auto upload'

path_dirs = [name for name in os.listdir(os.getcwd()) if
             os.path.isdir(os.path.join(os.getcwd(), name)) and not name.startswith(".")]

# git lfs migrate import --include="*.csv"
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

        with open(f'{jupyter_name}.md', 'r', encoding="utf-8") as file:
            header = file.readline()
            header = header.replace("#", "")
            header = header.strip()

        with open(f'{jupyter_name}.tex', 'r', encoding="utf-8") as file:
            filedata = file.read()
        for iter in range(min(len(OLD), len(NEW))):
            filedata = filedata.replace(OLD[iter], NEW[iter])
        filedata = make_title(filedata, header)
        with open(f'{jupyter_name}.tex', 'w', encoding="utf-8") as file:
            file.write(filedata)

        subprocess.call((f'pdflatex -interaction=batchmode {jupyter_name}.tex -output-format pdf'))
        try:
            os.remove(f"{header} - Отчёт Жидков А.А. R4136с.pdf")
        except:
            pass
        os.rename(f'{jupyter_name}.pdf', f"{header} - Отчёт Жидков А.А. R4136с.pdf")

        os.remove(f'{jupyter_name}.tex')
        os.remove(f'{jupyter_name}.log')
        os.remove(f'{jupyter_name}.aux')
        os.remove(f'{jupyter_name}.out')

    os.system(f'git reset')
    os.system(f'git add .')
    os.system(f'git commit -m \"{path_dir + " - " + commit_name}\"')
    os.system(f'git push')

    os.chdir(f'..')

os.system(f'git reset')
os.system(f'git add .')
os.system(f'git commit -m \"{commit_name}\"')
os.system(f'git push')

os.system(f'git lfs pull')
