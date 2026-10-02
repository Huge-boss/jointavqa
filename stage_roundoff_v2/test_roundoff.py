import ast,inspect,unittest
from unittest.mock import patch
from common_stage import *
import runtime
import roundoff_fix as fix


class RoundoffTests(unittest.TestCase):
    def test_boundary_is_accepted_but_actual_excess_and_invalid_intervals_rejected(self):
        for w in [[6.607,18.608],[5.037,17.038],[117.672,129.673],[0,12],[0,4]]:
            self.assertTrue(fix.valid_interval(w),w)
        for w in [[0,12.002],[-.001,11],[4,4],[8,7],[0,float('nan')],[0,float('inf')]]:
            self.assertFalse(fix.valid_interval(w),w)

    def test_ffmpeg_arguments_exactly_match_original_on_valid_interval(self):
        interval=[10.125,22.125]
        with patch.object(runtime,'probe',return_value=12),patch.object(runtime,'sha',return_value='mock'),patch.object(runtime.subprocess,'run') as run:
            expected=runtime.excerpt('source',interval,'dest');old_call=run.call_args
            observed=fix.excerpt('source',interval,'dest');new_call=run.call_args
        self.assertEqual(old_call,new_call);self.assertEqual(expected,observed)

    def test_ffmpeg_call_source_is_unchanged(self):
        def call_node(fn):
            tree=ast.parse(inspect.getsource(fn))
            return next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='subprocess')
        self.assertEqual(ast.dump(call_node(runtime.excerpt)),ast.dump(call_node(fix.excerpt)))


if __name__=='__main__':unittest.main()
