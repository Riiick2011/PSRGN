#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
at::Tensor Guass_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& center,
                                 const at::Tensor& gama,
                                 const float spatial_scale,
                                 const int pooled_height);

std::vector<at::Tensor> Guass_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& input,
                                  const at::Tensor& rois,
                                  const at::Tensor& center,
                                  const at::Tensor& gama,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int batch_size,
                                  const int channels,
                                  const int height);

// C++ interface
at::Tensor Guass_forward(const at::Tensor& input, // (bs,ch,t)
                                 const at::Tensor& rois, // (bs, start, end)
                                 const at::Tensor& center,
                                 const at::Tensor& gama,
                                 const int pooled_height){
    return Guass_forward_cuda(input, rois,center,gama, 1.0, pooled_height);
                                     }

std::vector<at::Tensor> Guass_backward(const at::Tensor& grad,
                                  const at::Tensor& input,
                                  const at::Tensor& rois,
                                  const at::Tensor& center,
                                  const at::Tensor& gama,
                                  const int pooled_height,
                                  const int batch_size,
                                  const int channels,
                                  const int height){
    return Guass_backward_cuda(grad, input, rois, center, gama, 1.0, pooled_height, batch_size, channels, height);
                                      }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &Guass_forward, "Guass forward (CUDA)");
  m.def("backward", &Guass_backward, "Guass backward (CUDA)");
}
