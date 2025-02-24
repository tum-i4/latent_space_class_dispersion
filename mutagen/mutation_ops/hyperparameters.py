import ast
from mut_base import TrainingMutation


class ChangeBatchSize(TrainingMutation):
    batch_size: int

    def visit_Call(self, node: ast.Call):
        assert self.current_function == "train"

        bs_arg = next(k for k in node.keywords if k.arg == 'batch_size')
        bs_arg.value = ast.Constant(self.batch_size)


class ChangeEpochs(TrainingMutation):
    epochs: int

    def visit_For(self, node: ast.For):
        assert isinstance(node.target, ast.Name)
        assert node.target.id == 'epoch'

        assert isinstance(node.iter, ast.Call)
        assert isinstance(node.iter.func, ast.Name)
        assert node.iter.func.id == 'range'

        node.iter.args[1] = ast.Constant(self.epochs)


class ChangeLearningRate(TrainingMutation):
    learning_rate: float

    def visit_Call(self, node: ast.Call):
        assert self.current_function == "train"

        bs_arg = next(k for k in node.keywords if k.arg == 'lr')
        bs_arg.value = ast.Constant(self.learning_rate)
