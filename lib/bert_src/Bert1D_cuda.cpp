#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
at::Tensor Bert_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const float spatial_scale,
                                 const int pooled_height);

at::Tensor Bert_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int batch_size,
                                  const int channels,
                                  const int height);

// C++ interface
at::Tensor Bert_forward(const at::Tensor& input, // (bs,ch,t)
                                 const at::Tensor& rois, // (bs, start, end)
                                 const int pooled_height){
    return Bert_forward_cuda(input, rois, 1.0, pooled_height);
                                     }

at::Tensor Bert_backward(const at::Tensor& grad,
                                  const at::Tensor& rois,
                                  const int pooled_height,
                                  const int batch_size,
                                  const int channels,
                                  const int height){
    return Bert_backward_cuda(grad, rois, 1.0, pooled_height, batch_size, channels, height);
                                      }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &Bert_forward, "Bert forward (CUDA)");
  m.def("backward", &Bert_backward, "Bert backward (CUDA)");
}
