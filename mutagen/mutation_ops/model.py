import ast
from mut_base import ModelMutation


def _assign_call_to_self(node: ast.Assign, member: str) -> ast.Attribute:
    assert len(node.targets) == 1

    name = node.targets[0]
    assert isinstance(name, ast.Attribute)
    assert isinstance(name.value, ast.Name)
    assert name.value.id == 'self'

    assert name.attr == member

    assert isinstance(node.value, ast.Call)

    assert isinstance(node.value.func, ast.Attribute)
    return node.value.func


class ChangeActivation(ModelMutation):
    member: str
    activation: str

    def visit_Assign(self, node: ast.Assign):
        assert self.current_function == "__init__"

        func = _assign_call_to_self(node, self.member)
        assert isinstance(func.value, ast.Name)
        assert func.value.id == 'nn'

        func.attr = self.activation


class RemoveBatchNorm(ModelMutation):
    member: str

    def visit_Assign(self, node: ast.Assign):
        assert self.current_function == "__init__"

        func = _assign_call_to_self(node, self.member)
        assert isinstance(func.value, ast.Name)
        assert func.value.id == 'nn'
        assert func.attr.startswith('BatchNorm')

        func.attr = 'Identity'


class ReplaceMemberCall(ModelMutation):
    current: str
    replacement: str

    def visit_Attribute(self, node: ast.Attribute):
        assert isinstance(node.value, ast.Name)
        assert node.value.id == 'self'

        assert node.attr == self.current
        node.attr = self.replacement


class ChangeKeywordInAssign(ModelMutation):
    member: str
    keyword: str
    value: str

    def visit_Assign(self, node: ast.Assign):
        assert self.current_function == "__init__"

        func = _assign_call_to_self(node, self.member)
        assert isinstance(func.value, ast.Name)

        assert isinstance(node.value, ast.Call)
        keyword = next(k for k in node.value.keywords if k.arg == self.keyword)
        keyword.value = ast.Constant(ast.literal_eval(self.value))
