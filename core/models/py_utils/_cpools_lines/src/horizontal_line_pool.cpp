#include <torch/torch.h>

#include <vector>

std::vector<at::Tensor> pool_forward(
    at::Tensor input
) {
    // Initialize output
    at::Tensor output = at::zeros_like(input);
    output.copy_(input);

    auto maxVal = at::max(output, 3, true);
    auto ret = std::get<0>(maxVal);

    return { 
        ret
    };
}

std::vector<at::Tensor> pool_backward(
    at::Tensor input,
    at::Tensor grad_output
) {
    auto output = at::zeros_like(input);

    int32_t batch   = input.size(0);
    int32_t channel = input.size(1);
    int32_t height  = input.size(2);
    int32_t width   = input.size(3);

    for (int32_t ind = 0; ind < width; ++ind) {
	auto slice = at::slice(output, 3, ind, ind + 1);
	slice.copy_(grad_output);	
    }

    return {
        output
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward", &pool_forward, "Horizontal Line Pool Forward", 
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "backward", &pool_backward, "Horizontal Line Pool Backward", 
        py::call_guard<py::gil_scoped_release>()
    );
}
