import ast
from typing import List
from mut_base import TrainingMutation


def _str2keywords(extra_args: str) -> List[ast.keyword]:
    extra_arg_pairs = [value.split(':') for value in extra_args.split("|")]
    extra_arg_pairs = [(name, ast.literal_eval(value)) for name, value in extra_arg_pairs]

    extra_keywords = [ast.keyword(arg=name, value=ast.Constant(value)) for name, value in extra_arg_pairs]

    return extra_keywords


class ChangeLoss(TrainingMutation):
    loss: str

    def visit_Assign(self, node: ast.Assign):
        assert self.current_function == "train"

        assert len(node.targets) == 1

        name = node.targets[0]
        assert isinstance(name, ast.Name)
        assert name.id == "criterion"

        assert isinstance(node.value, ast.Call)
        assert isinstance(node.value.func, ast.Attribute)

        assert isinstance(node.value.func.value, ast.Name)
        assert node.value.func.value.id == "nn"

        node.value.func.attr = self.loss


class ChangeOptimizer(TrainingMutation):
    optimizer: str
    extra_arg: str = ""

    def visit_Assign(self, node: ast.Assign):
        assert self.current_function == "train"

        assert len(node.targets) == 1

        name = node.targets[0]
        assert isinstance(name, ast.Name)
        assert name.id == "optimizer"

        assert isinstance(node.value, ast.Call)
        assert isinstance(node.value.func, ast.Attribute)

        assert isinstance(node.value.func.value, ast.Name)
        assert node.value.func.value.id == "optim"

        assert len(node.value.args) == 1

        node.value.func.attr = self.optimizer
        node.value.keywords = [k for k in node.value.keywords if k.arg in {"lr"}]
        node.value.keywords.extend(_str2keywords(self.extra_arg))


class ChangeScheduler(TrainingMutation):
    scheduler: str
    extra_arg: str = ""

    def visit_Assign(self, node: ast.Assign):
        assert self.current_function == "train"

        assert len(node.targets) == 1

        name = node.targets[0]
        assert isinstance(name, ast.Name)
        assert name.id == "scheduler"

        assert isinstance(node.value, ast.Call)
        assert isinstance(node.value.func, ast.Attribute)

        assert isinstance(node.value.func.value, ast.Name)
        assert node.value.func.value.id == "lr_scheduler"

        assert len(node.value.args) == 1

        arg = node.value.args[0]
        assert isinstance(arg, ast.Name)
        assert arg.id == "optimizer"

        node.value.func.attr = self.scheduler
        node.value.keywords = [k for k in node.value.keywords if k.arg in {"verbose"}]
        node.value.keywords.extend(_str2keywords(self.extra_arg))

