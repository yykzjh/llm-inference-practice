#pragma once

#include <torch/torch.h>

void scaled_multiply(torch::Tensor& data, float scale);
