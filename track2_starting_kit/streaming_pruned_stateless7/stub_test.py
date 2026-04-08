import sys, signal, types
from pathlib import Path

def t(sig, frame):
    print('TIMEOUT', flush=True)
    sys.exit(1)
signal.signal(signal.SIGALRM, t)
signal.alarm(20)

_ICEFALL = Path('icefall').resolve()
_RECIPE = _ICEFALL / 'egs/librispeech/ASR/pruned_transducer_stateless7_streaming'
sys.path.insert(0, str(_ICEFALL))
sys.path.insert(0, str(_RECIPE))

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load('weights/libri_giga/data/lang_bpe_500/bpe.model')
print('spm ok', flush=True)

def stub(name, **a):
    m = types.ModuleType(name)
    for k, v in a.items():
        setattr(m, k, v)
    return m

stubs = {
    'lhotse': stub('lhotse'),
    'lhotse.cut': stub('lhotse.cut', Cut=object),
    'lhotse.dataset': stub('lhotse.dataset'),
    'lhotse.dataset.sampling': stub('lhotse.dataset.sampling'),
    'lhotse.dataset.sampling.base': stub('lhotse.dataset.sampling.base', CutSampler=object),
    'lhotse.utils': stub('lhotse.utils', fix_random_seed=lambda *a, **k: None, uuid4=lambda: 'x'),
    'icefall': stub('icefall'),
    'icefall.dist': stub('icefall.dist', cleanup_dist=lambda: None, setup_dist=lambda: None),
    'icefall.env': stub('icefall.env', get_env_info=lambda: {}),
    'icefall.err': stub('icefall.err', raise_grad_scale_is_too_small_error=lambda *a, **k: None),
    'icefall.hooks': stub('icefall.hooks', register_inf_check_hooks=lambda *a, **k: None),
    'icefall.checkpoint': stub('icefall.checkpoint', load_checkpoint=lambda *a, **k: {}),
    'icefall.utils': stub('icefall.utils', AttributeDict=dict, add_sos=lambda *a, **k: None,
                          torch_autocast=lambda *a, **k: None, get_texts=lambda *a, **k: [],
                          setup_logger=lambda *a, **k: None, str2bool=lambda v: v,
                          make_pad_mask=lambda *a, **k: None),
}
for k, v in stubs.items():
    sys.modules[k] = v
print('stubs injected', flush=True)

print('decode_stream...', flush=True)
from decode_stream import DecodeStream
print('ok', flush=True)

print('streaming_beam_search...', flush=True)
from streaming_beam_search import greedy_search, modified_beam_search
print('ok', flush=True)

print('zipformer...', flush=True)
from zipformer import stack_states, unstack_states
print('ok', flush=True)

print('train...', flush=True)
from train import add_model_arguments, get_params, get_transducer_model
print('ok', flush=True)

print('ALL OK', flush=True)
