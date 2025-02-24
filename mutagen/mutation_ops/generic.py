import ast
from mut_base import TrainingMutation


# TODO: make this work for training, model and eval
class ChangeKeyword(TrainingMutation):
    function: str = ""
    keyword: str
    value: str

    def visit_Call(self, node: ast.Call):
        if self.function:
            assert self.current_function == self.function

        bs_arg = next(k for k in node.keywords if k.arg == self.keyword)
        bs_arg.value = ast.Constant(ast.literal_eval(self.value))


class RemoveCall(TrainingMutation):
    call: str

    def visit_Call(self, node: ast.Call):
        assert isinstance(node.func, ast.Attribute)
        assert isinstance(node.func.value, ast.Name)

        # TODO: support arbitrary long attributes
        var, method = self.call.split(".")
        assert node.func.value.id == var
        assert node.func.attr == method

        return ast.Str(self.call)
