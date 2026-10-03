Frozen V13 full-dataset evaluation, authorized2026-10-03.
Joint2853 / AV-Speaker3212, two models12130 predictions. Reuse2426 completed V13 dev20 predictions and all existing native baselines; run only9704 missing method predictions.
method.py and ledger_grammar.py byte-identical to V13. No V14/markers, no baseline calls, no new pilot or algorithm tuning. Same GPU0/1/3 stage schedule. Independent labels/scoring only after stages.
Report dev20/complement80/full separately; original split was question-level and baseline was previously audited, so not a pristine heldout claim. All errors/missing count in denominator. Retain all old frozen files.
