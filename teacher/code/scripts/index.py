from collections import defaultdict
from json import load
from pywebio.input import input, FLOAT
from pywebio.output import *
import pywebio
import yaml
import os
print(pywebio.STATIC_PATH)

def main():
    try:
        with open('lib/demo.yaml') as f:
            d = yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)
    # url_list = [d[i] for i in d]
    # for i in url_list:
    #     pywebio.output.put_html(f'<video controls="controls" src="{i}"></video>'.format(url=i))

    # scope = pywebio.output.put_scope('scope')
    dd = defaultdict(list)
    for i in d['support']:
        html = pywebio.output.put_html(f'<video controls="controls" width="250" height="250" src="{i}"></video>')
        c = i.split('/')[-2]
        dd[c].append(html)
    # pywebio.output.put_html(f'<video controls="controls" src="{i}"></video>'.format(url=i))
    
    target = d['target'][0]
    c = target.split('/')[-2]
    target = f'<video controls="controls" width="250" height="250" src="{target}"></video>'
    out = [[span('action\support'), 'support1', 'support2',
                'support3', 'support4', 'support5']] + [[key] + dd[key] for key in dd]
    pywebio.output.put_text(c).style("margin-left:750px")
    pywebio.output.put_html(target).style("margin-left:700px")
    pywebio.session.set_env(output_max_width="1600px")
    with use_scope("scope") as scope:
        put_table(out)


pywebio.start_server(main, port=8089)
