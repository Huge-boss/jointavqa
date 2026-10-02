"""Bound candidate syntax and media address space; no labels or scoring."""
class LedgerGrammar:
    def __init__(self,tok,eos_ids,frame_count):
        self.tok=tok
        self.eos=list(eos_ids) if isinstance(eos_ids,(list,tuple)) else [eos_ids]
        self.eos=[x for x in self.eos if x is not None];assert self.eos
        assert 0<frame_count<=128
        enc=lambda s:tok.encode(s,add_special_tokens=False)
        self.spec=[('CANDIDATE',enc(''),{s:enc(s) for s in 'ABCD'},None),
                   ('EVIDENCE',enc('\nEVIDENCE: '),None,48),
                   ('ANCHOR',enc('\nANCHOR: '),{s:enc(s) for s in ['NONE']+[f'F{i:03d}' for i in range(frame_count)]},None),
                   ('RELATION',enc('\nRELATION: '),{s:enc(s) for s in ['BEFORE','AROUND','AFTER','GLOBAL']},None)]
        self.field=0;self.start=0;self.base=None;self.values={};self.caps={};self.last_generated=[]

    def allowed(self,g,preferred):
        self.last_generated=list(g)
        while self.field<len(self.spec):
            name,prefix,choices,limit=self.spec[self.field]
            offset=len(g)-self.start
            if offset<len(prefix):return [prefix[offset]]
            begin=self.start+len(prefix);tail=g[begin:]
            if choices is not None:
                for value,seq in choices.items():
                    if tail==seq:
                        self.values[name]=value;self.field+=1;self.start=len(g);break
                else:
                    matches=[seq for seq in choices.values() if seq[:len(tail)]==tail]
                    assert matches,'Invalid ledger enum prefix'
                    return sorted({seq[len(tail)] for seq in matches})
                continue
            body=self.tok.decode(tail,skip_special_tokens=True).strip()
            newline='\n' in self.tok.decode([preferred],skip_special_tokens=False)
            if len(tail)>=limit or (tail and (preferred in self.eos or newline)):
                self.values[name]=body or 'UNKNOWN';self.caps[name]=len(tail)>=limit
                self.field+=1;self.start=len(g);continue
            return None if tail else ('exclude',self.eos)
        return [self.eos[0]]

    def __call__(self,input_ids,scores):
        import torch
        assert input_ids.shape[0]==1
        if self.base is None:self.base=input_ids.shape[1]
        allowed=self.allowed(input_ids[0,self.base:].tolist(),int(scores[0].argmax()))
        if allowed is None:return scores
        if isinstance(allowed,tuple):
            out=scores.clone();out[:,allowed[1]]=-float('inf');return out
        out=torch.full_like(scores,-float('inf'));out[:,allowed]=scores[:,allowed];return out

    def fields(self,generated):
        assert self.field==len(self.spec) and generated[-1] in self.eos,'Incomplete candidate ledger'
        g=self.last_generated
        assert generated[-len(g)-1:-1]==g
        return dict(self.values,field_token_caps=self.caps,syntax='candidate_then_evidence_v13')
