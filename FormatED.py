import argparse
import pyonmttok
import edit_distance


class FormatWithEditDist():

    def __init__(self, onmttok, BAR='║', BEG='《', END='》'):
        self.onnmttok = onmttok
        self.BAR = BAR
        self.BEG = "" #BEG                                                                                                                                                                                                                                                       
        self.END = "" #END                                                                                                                                                                                                                                                       

    def __call__(self, src_txt, tgt_txt):
        src = onmttok(src_txt)
        tgt = onmttok(tgt_txt)
        sm = edit_distance.SequenceMatcher(a=src, b=tgt)
        out = []
        for opcode in sm.get_opcodes():
            code, src_beg, src_end, tgt_beg, tgt_end = opcode
            if code == 'equal':
                out.append(' '.join(src[src_beg:src_end]))
            else:
                out.append(self.BEG + ' '.join(src[src_beg:src_end]) + self.BAR + ' '.join(tgt[tgt_beg:tgt_end]) + self.END)
        return ' '.join(out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", default=None, type=str, help="hypotheses file")
    parser.add_argument("ref", default=None, type=str, help="references file")
    parser.add_argument("--tok", default="aggressive", type=str, help="onmt tokenization type (aggressive)")
    args = parser.parse_args()
    
    onmttok = pyonmttok.Tokenizer(args.tok, joiner_annotate=False)
    formatWithED = FormatWithEditDist(onmttok)

    with open(args.hyp, 'r') as fd:
        hyp = fd.readlines()
    with open(args.ref, 'r') as fd:
        ref = fd.readlines()
    assert(len(hyp) == len(ref))
    
    for i in range(len(hyp)):
        print(formatWithED(hyp[i], ref[i]))
