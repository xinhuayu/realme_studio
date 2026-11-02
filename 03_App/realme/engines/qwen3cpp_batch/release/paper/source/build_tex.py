"""report.md -> report.tex (arXiv-style article) -> report.pdf."""
import re, subprocess, pathlib
here = pathlib.Path(__file__).parent
md = (here/'report.md').read_text(encoding='utf-8')
front, body = md.split('---', 2)[1], md.split('---', 2)[2]
title = re.search(r'title: "(.*)"', front).group(1)
author = re.search(r'author: "(.*)"', front).group(1)
date = re.search(r'date: "(.*)"', front).group(1)
abstract, rest = body.split('## 1 Introduction', 1)
abstract = abstract.replace('## Abstract', '').strip()
rest = '## 1 Introduction' + rest
# headings: drop manual numbers, LaTeX numbers them in the same order
rest = re.sub(r'^(#{2,3}) \d+(\.\d+)? ', r'\1 ', rest, flags=re.M)
# unnumbered back matter
for h in ('Declaration on the use of AI tools', 'References'):
    rest = rest.replace(f'## {h}', f'## {h} {{-}}')
# table captions: LaTeX numbers them
rest = re.sub(r'^Table: Table \d+\. ', 'Table: ', rest, flags=re.M)
def conv(text):
    return subprocess.run(['pandoc', '-f', 'markdown+pipe_tables+implicit_figures+autolink_bare_uris', '-t', 'latex',
                           '--top-level-division=section', '--shift-heading-level-by=-1'],
                          input=text, capture_output=True, text=True, check=True).stdout
fig_md = ''
m = re.search(r'^!\[.*$', abstract, flags=re.M)
if m:
    fig_md, abstract = m.group(0), abstract.replace(m.group(0), '').strip()
abs_tex = conv(abstract)
fig_tex = conv(fig_md).replace('\\begin{figure}', '\\begin{figure}[htbp]') if fig_md else ''
abs_tex = re.sub(r'^(Background|Methods|Results|Conclusions)\. ', r'\\textit{\1.} ', abs_tex, flags=re.M)
body_tex = conv(rest)
body_tex = body_tex.replace('\\begin{figure}', '\\begin{figure}[tbp]')
body_tex = re.sub(r'\\url\{', r'\\url{', body_tex)
tex = r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{mathptmx}
\usepackage{microtype}
\usepackage{graphicx}
\setkeys{Gin}{keepaspectratio}
\usepackage{booktabs,longtable,array,calc}
\usepackage[font=small,labelfont=bf,labelsep=period]{caption}
\usepackage[hidelinks]{hyperref}
\usepackage{xurl}
\setlength{\parskip}{0.35em}
\setlength{\parindent}{0pt}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\def\LTcaptype{table}
\renewcommand{\arraystretch}{1.15}
\setlength{\LTpre}{0.6em}\setlength{\LTpost}{0.6em}
\title{\Large\textbf{""" + title + r"""}}
\author{""" + author + r"""}
\date{""" + date + r"""}
\begin{document}
\maketitle
\begin{abstract}
\noindent """ + abs_tex + r"""
\end{abstract}
""" + fig_tex + body_tex + r"""
\end{document}
"""
(here/'report.tex').write_text(tex, encoding='utf-8')
for _ in range(2):
    r = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'report.tex'], cwd=here, capture_output=True, text=True)
print(r.stdout[-1500:])
