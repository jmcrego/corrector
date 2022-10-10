import copy
import jiwer
import pyonmttok

class wer():
    def __init__(self, onmttok=None):
        self.onmttok = onmttok

    def __call__(self, hyps, refs):
        hyps = copy.deepcopy(hyps)
        refs = copy.deepcopy(refs)
        assert len(refs) == len(hyps)
        if self.onmttok is not None:
            n_hyp_words = 0
            for i,l in enumerate(hyps):
                t = self.onmttok(l)
                n_hyp_words += len(t)
                hyps[i] = ' '.join(t)

            n_ref_words = 0
            for i,l in enumerate(refs):
                t = self.onmttok(l)
                n_ref_words += len(t)
                refs[i] = ' '.join(t)
                
        return 100.0 * jiwer.wer(refs, hyps), n_hyp_words, n_ref_words
        
if __name__ == '__main__':

    import sys
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=str, help='reference file/s', required=True)
    parser.add_argument('--hyp', type=str, default=None, help='hypothesis file')
    parser.add_argument("--onmttok", type=str, default=None, help="onmt tokenization: conservative OR aggressive (None)")
    args = parser.parse_args()

    onmttok = None
    if args.onmttok is not None:
        onmttok = pyonmttok.Tokenizer(args.onmttok, joiner_annotate=False)
        
    with open(args.ref, 'r') as fd:
        refs = fd.readlines()

    if args.hyp is not None:
        with open(args.hyp, 'r') as fd:
            hyps = fd.readlines()
    else:
        hyps = []
        for l in sys.stdin:
            hyps.append(l.rstrip())
            
    score = wer(onmttok)
    s, nh, nr = score(hyps,refs)
    print("WER: {:.2f} % #hyp={} #ref={} onmttok={}".format(s,nh,nr,args.onmttok))
