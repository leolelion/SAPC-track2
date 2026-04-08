import sys
sys.path.insert(0, '.')
from model import Model
import numpy as np

m = Model()
m.set_partial_callback(lambda t: None)
m.reset()
r = m.accept_chunk(np.zeros(1600, dtype=np.float32))
print("chunk ok:", repr(r))
f = m.input_finished()
print("finished ok:", repr(f))
print("ALL OK")
