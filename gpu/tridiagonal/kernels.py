from pycuda import autoinit
import pycuda.compiler as compiler
import numpy as np
import jinja2
import os

def render_kernel(template_filename, out_filename, **kwargs):
    src_dir = os.path.dirname(__file__)
    with open(src_dir + '/' + template_filename) as f:
        kernel_template = f.read()
    tpl = jinja2.Template(kernel_template)
    kernel = tpl.render(**kwargs)
    with open(src_dir + '/' + out_filename, 'w') as f:
        f.write(kernel)

def get_funcs(filename, *args):
    src_dir = os.path.dirname(__file__)
    with open(src_dir + '/' + filename) as f:
        kernel_source = f.read()
    module = compiler.SourceModule(kernel_source, options=['-lineinfo', '-O2'], arch='sm_35')
    
    funcs = []
    for func_name in args:
        funcs.append(module.get_function(func_name))
    return funcs

