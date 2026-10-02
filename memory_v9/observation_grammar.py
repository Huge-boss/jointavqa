"""Constrain tool record syntax only; never constrain answer content or read labels."""
class ObservationGrammar:
    def __init__(self,tokenizer,eos_ids):
        self.tok=tokenizer
        self.eos=list(eos_ids) if isinstance(eos_ids,(list,tuple)) else [eos_ids]
        self.eos=[v for v in self.eos if v is not None];assert self.eos
        enc=lambda s:tokenizer.encode(s,add_special_tokens=False)
        self.names=['VISUAL','AUDIO','LINK','NEED']
        self.limits=[40,40,24,24]
        self.prefixes=[enc(('' if i==0 else '\n')+k+': ') for i,k in enumerate(self.names)]
        self.focus_prefix=enc('\nFOCUS: ')
        self.choices={s:enc(s) for s in ['UNKNOWN']+[str((i+.5)/8) for i in range(8)]}
        self.base=None;self.field=0;self.start=0;self.body_start=None;self.bounds={};self.caps={}
        self.focus=None;self.last_generated=None

    def allowed(self,g,preferred):
        self.last_generated=list(g)
        while self.field<4:
            prefix=self.prefixes[self.field]
            offset=len(g)-self.start
            if offset<len(prefix):return [prefix[offset]]
            begin=self.start+len(prefix);free=len(g)-begin
            # Respect a native end/newline after at least one body token. The next field is forced.
            newline='\n' in self.tok.decode([preferred],skip_special_tokens=False)
            body=self.tok.decode(g[begin:],skip_special_tokens=True).strip()
            sentence_done=free>2 and body.endswith(('.', '!', '?'))
            if free>=self.limits[self.field] or (free>0 and (preferred in self.eos or newline or sentence_done)):
                name=self.names[self.field];self.bounds[name]=(begin,len(g));self.caps[name]=free>=self.limits[self.field]
                self.field+=1;self.start=len(g);continue
            return None if free else ('exclude',self.eos)
        offset=len(g)-self.start
        if offset<len(self.focus_prefix):return [self.focus_prefix[offset]]
        tail=g[self.start+len(self.focus_prefix):]
        for value,seq in self.choices.items():
            if tail==seq:self.focus=value;return [self.eos[0]]
        matches=[seq for seq in self.choices.values() if seq[:len(tail)]==tail]
        assert matches,'Tool format left allowed focus choices'
        return sorted({seq[len(tail)] for seq in matches})

    def __call__(self,input_ids,scores):
        import torch
        assert input_ids.shape[0]==1
        if self.base is None:self.base=input_ids.shape[1]
        g=input_ids[0,self.base:].tolist();allowed=self.allowed(g,int(scores[0].argmax()))
        if allowed is None:return scores
        if isinstance(allowed,tuple):
            out=scores.clone();out[:,allowed[1]]=-float('inf');return out
        out=torch.full_like(scores,-float('inf'));out[:,allowed]=scores[:,allowed];return out

    def fields(self,generated):
        assert self.focus is not None and len(self.bounds)==4,'Incomplete tool record'
        g=self.last_generated
        assert generated[-1] in self.eos and generated[-len(g)-1:-1]==g
        out={k:self.tok.decode(g[a:b],skip_special_tokens=True).strip() or 'UNKNOWN' for k,(a,b) in self.bounds.items()}
        out.update(FOCUS=self.focus,field_token_caps=self.caps,syntax='tool_observation_v9_no_answer_field')
        return out
