"""Constrained field syntax inside one greedy generation; no labels or extra model calls."""


class MemoryGrammar:
    def __init__(self,tokenizer,eos_ids,evidence_limit=48):
        self.tok=tokenizer
        self.eos=list(eos_ids) if isinstance(eos_ids,(tuple,list)) else [eos_ids]
        self.eos=[v for v in self.eos if v is not None]
        assert self.eos
        enc=lambda x:tokenizer.encode(x,add_special_tokens=False)
        self.prefix=enc('EVIDENCE=');self.seek_prefix=enc('\nSEEK=');self.answer_prefix=enc('\nANSWER=')
        self.seek={k:enc(k) for k in ['NONE']+[str(i) for i in range(8)]}
        self.answers={k:enc(k) for k in 'ABCD'}
        assert all(v for v in list(self.seek.values())+list(self.answers.values()))
        self.limit=evidence_limit;self.base=None;self.body_end=None
        self.seek_value=None;self.answer_value=None;self.evidence_limited=False;self.last_generated=None

    @staticmethod
    def choice(tokens,options):
        for value,seq in options.items():
            if tokens[:len(seq)]==seq:return value,len(seq),None
        matches=[seq for seq in options.values() if seq[:len(tokens)]==tokens]
        assert matches,'Generation left the declared grammar'
        return None,None,sorted({seq[len(tokens)] for seq in matches})

    def allowed(self,g,preferred):
        self.last_generated=list(g)
        if len(g)<len(self.prefix):return [self.prefix[len(g)]]
        if self.body_end is None:
            free=len(g)-len(self.prefix)
            if free>=self.limit or (free>0 and preferred in self.eos):
                self.body_end=len(g);self.evidence_limited=free>=self.limit
            else:
                # At least one actual evidence token is needed; otherwise let the native model speak freely.
                return None if free else ('exclude',self.eos)
        offset=len(g)-self.body_end
        if offset<len(self.seek_prefix):return [self.seek_prefix[offset]]
        tail=g[self.body_end+len(self.seek_prefix):]
        value,n,allowed=self.choice(tail,self.seek)
        if allowed is not None:return allowed
        self.seek_value=value;tail=tail[n:]
        if len(tail)<len(self.answer_prefix):return [self.answer_prefix[len(tail)]]
        tail=tail[len(self.answer_prefix):]
        value,n,allowed=self.choice(tail,self.answers)
        if allowed is not None:return allowed
        self.answer_value=value
        return [self.eos[0]]

    def __call__(self,input_ids,scores):
        import torch
        assert input_ids.shape[0]==1,'Grammar supports greedy batch1 only'
        if self.base is None:self.base=input_ids.shape[1]
        g=input_ids[0,self.base:].tolist()
        allowed=self.allowed(g,int(scores[0].argmax()))
        if allowed is None:return scores
        if isinstance(allowed,tuple):
            out=scores.clone();out[:,allowed[1]]=-float('inf');return out
        out=torch.full_like(scores,-float('inf'));out[:,allowed]=scores[:,allowed]
        return out

    def fields(self,generated):
        assert self.body_end is not None and self.seek_value is not None and self.answer_value is not None,'Incomplete constrained memory'
        # inputs_embeds generation can retain a virtual BOS in its returned tensor.
        # The processor sees only tokens generated after the first decoding step.
        g=self.last_generated
        assert generated[-1] in self.eos and generated[-len(g)-1:-1]==g,'Returned tokens differ from grammar trace'
        evidence=self.tok.decode(g[len(self.prefix):self.body_end],skip_special_tokens=True).strip()
        return dict(evidence=evidence,seek=self.seek_value,answer=self.answer_value,
                    evidence_limited=self.evidence_limited,syntax='evidence48tokens_seek_NONE_or_bin0to7_answer_ABCD')
