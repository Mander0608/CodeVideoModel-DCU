"""Reduced end-to-end checkpoint test, NOT the paper's quality settings.

256x160, 22 frames, 2 updates, proxy present for only the first update.
All original checkpoint parameters are used, without quantization.
"""
import os, runpy
import cvm
cvm.WIDTH, cvm.HEIGHT, cvm.FRAMES = 256, 160, 22
cvm.STEPS, cvm.KEEP_STEPS = 2, 1
os.environ['CVM_SMOKE_TEST']='1'
runpy.run_module('run_dcu',run_name='__main__')
