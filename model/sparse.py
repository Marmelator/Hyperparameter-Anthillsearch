import torch
from torch import nn
from torch.autograd import Function


class SparseLinearFunction(Function):

    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        output = input.mm(weight.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)

        return grad_input, grad_weight


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, connectivity_matrix):
        super(SparseLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)).type(torch.FloatTensor)
        nn.init.xavier_uniform_(self.weight)
        if self.weight.shape != connectivity_matrix.shape:
            raise AttributeError("Connectivity Matrix must be same size as weights, got {} instead"
                                 .format(connectivity_matrix.shape))
        self.mask = connectivity_matrix

    def forward(self, input):
        masked_weight = self.weight * self.mask
        return SparseLinearFunction.apply(input, masked_weight)


