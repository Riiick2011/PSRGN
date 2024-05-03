// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
// Modifies by Frost for 1D ussage
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
#include <cuda.h>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ T linear_interpolate(const T* bottom_data,
    const int height,
    T t,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (t < -1.0 || t > height) {
    //empty
    return 0;
  }

  if (t <= 0) t = 0;

  int t_low = (int) t;


  // do linear interpolation
  T val = bottom_data[t_low];
  // printf("Check Linear Interpolate: w1=%f, v1=%f, w2=%f, v2=%f \n", w1, v1, w2, v2);
  return val;
}

template <typename T>
__global__ void Super1DForward(const int nthreads, const T* bottom_data,const T* center_data,
    const T* gama_data,
    const T spatial_scale, const int channels,
    const int height,
    const int num_rois,
    const int pooled_height,
    const T* bottom_rois, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pt) is an element in the pooled output
    int pt = index % pooled_height;
    int c = (index / pooled_height) % channels;
    int n_roi = (index / pooled_height / channels) % num_rois;
    int n_g =index / pooled_height / channels/ num_rois;

    // printf("Debug Main Loop: get pt, c, n are %d, %d, %d \n", pt, c, n);

    const T* offset_bottom_rois = bottom_rois + n_roi * 3;
    int roi_batch_ind = offset_bottom_rois[0];
    const T* offset_center = center_data + n_g;
    const T* offset_gama = gama_data + n_g;


    // Do not using rounding; this implementation detail is critical
    T roi_start = offset_bottom_rois[1] * spatial_scale;
    T roi_end = offset_bottom_rois[2] * spatial_scale;
    T center_u=offset_center[0];
    T gama_u=offset_gama[0];
    // printf("Debug roi boundary: w1,  w2,  is  %f, %f \n", roi_start,roi_end,);

    // Force malformed ROIs to be 1x1
    T roi_height = max(roi_end- roi_start, (T)1.);
    T bin_size = static_cast<T>(roi_height) / static_cast<T>(pooled_height);

    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid = ceil(roi_height / pooled_height); // e.g., = 2

    // We do average (integral) pooling inside a bin
    T output_val = 0.;
    T output_weight = 0.;
    for (int it = 0; it < roi_bin_grid; it ++) // e.g., it = 0, 1
    {
      const T t = roi_start + pt * bin_size + static_cast<T>(it) ; // e.g., 0.5, 1.5
      int t_low = (int) t;
      T val = offset_bottom_data[t_low];
      T x_id = static_cast<T>(it) / static_cast<T>(roi_bin_grid) ;
      T center_u_0= center_u;
      T dela = max(0.0000001,(x_id-center_u_0)*(x_id-center_u_0)+gama_u*gama_u);
      T weight=1.0/3.1415926535898/dela*gama_u ;

      // T val = linear_interpolate(offset_bottom_data, height, t, index);
      // printf("Debug linear_interpolate: input=height:%d, t:%f, ... ; output=val:%f \n", height, t, val);
      output_val += val*weight;
      output_weight += weight;

    }
    output_val /= output_weight;

    top_data[index] = output_val;
  }
}


template <typename T>
__device__ void linear_interpolate_gradient(
    const int height, 
    T t,
    T & w1, T & w2,
    int & t_low, int & t_high, 
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (t < -1.0 || t > height) {
    //empty
    w1 = w2 = 0.;
    t_low = t_high = -1;
    return;
  }

  if (t <= 0) t = 0;

  t_low = (int) t;

  if (t_low >= height - 1) {
    t_high = t_low = height - 1;
    t = (T) t_low;
  } else {
    t_high = t_low + 1;
  }

  T lt = t - t_low;
  T ht = 1. - lt;

  // T val = (w1 * v1 + w2 * v2);
  // T w1 = ht, w2 = lt;
  w1 = ht , w2 = lt;

  return;
}

template <typename T>
__global__ void Super1DBackwardFeature(const int nthreads, const T* top_diff,
    const T* bottom_data,
    const T* center_data,
    const T* gama_data,
    const int num_rois, const T spatial_scale,
    const int channels, const int height,
    const int pooled_height,
    T* bottom_diff,
    T* diff_center,
    T* diff_gama,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, pt) is an element in the pooled output
    int pt = (index ) % pooled_height;
    int c = (index / pooled_height) % channels;
    int n_roi = (index / pooled_height / channels) % num_rois;
    int n_g =index / pooled_height / channels/ num_rois;

    const T* offset_bottom_rois = bottom_rois + n_roi * 3;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not using rounding; this implementation detail is critical
    T roi_start= offset_bottom_rois[1] * spatial_scale;
    T roi_end= offset_bottom_rois[2] * spatial_scale;

    // Force malformed ROIs to be 1x1
    T roi_height = max(roi_end- roi_start, (T)1.);
    T bin_size = static_cast<T>(roi_height) / static_cast<T>(pooled_height);

    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height;
    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height;
    T* offset_diff_center = diff_center +n_g;
    T* offset_diff_gama = diff_gama + n_g;
    const T* offset_center = center_data + n_g;
    const T* offset_gama = gama_data + n_g;

    const T top_diff_this_bin = top_diff[index];
    T center_u=offset_center[0];
    T gama_u=offset_gama[0];


    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid=ceil(roi_height / pooled_height); // e.g., = 2

    // We do average (integral) pooling inside a bin
    T output_weight = 0.;
    for (int it = 0; it < roi_bin_grid; it ++) // e.g., it = 0, 1
    {
      T x_id = static_cast<T>(it) / static_cast<T>(roi_bin_grid) ;
      T center_u_0= center_u;
      T dela = max(0.0000001,(x_id-center_u_0)*(x_id-center_u_0)+gama_u*gama_u);
      T weight=1.0/3.1415926535898/dela*gama_u ;
      output_weight += weight;
    }

    for (int it = 0; it < roi_bin_grid; it ++) // e.g., iy = 0, 1
    {
      const T df_t = roi_start+ pt * bin_size+ static_cast<T>(it + .0f); // e.g., 0.5, 1.5

      //T w1, w2;
      int df_t_low = (int) df_t;

      //linear_interpolate_gradient(height, t, w1, w2, t_low, t_high, index);
      T df_x_id = static_cast<T>(it) / static_cast<T>(roi_bin_grid) ;
      T df_center_u_0= center_u ;
      T df_dela = max(0.0000001,(df_x_id-df_center_u_0)*(df_x_id-df_center_u_0)+gama_u*gama_u);
      T df_weight= 1.0/3.1415926535898/df_dela*gama_u ;
      T val = offset_bottom_data[df_t_low];
      T df_dela_3 = max(0.0000001,((df_x_id-df_center_u_0)*(df_x_id-df_center_u_0)+gama_u*gama_u)*((df_x_id-df_center_u_0)*(df_x_id-df_center_u_0)+gama_u*gama_u));

      T g = top_diff_this_bin / output_weight * df_weight;
      T g_center = top_diff_this_bin / output_weight *val/3.1415926535898*(df_x_id-df_center_u_0)*2/df_dela_3;
      T g_gama = top_diff_this_bin / output_weight *val/3.1415926535898/df_dela_3*((df_x_id-df_center_u_0)*(df_x_id-df_center_u_0)-gama_u*gama_u);

      if (df_t_low >= 0)
      {
          atomicAdd(offset_bottom_diff + df_t_low, static_cast<T>(g));
          atomicAdd(offset_diff_center, static_cast<T>(g_center));
          atomicAdd(offset_diff_gama, static_cast<T>(g_gama));
      } // if
    } // it
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward


at::Tensor Super_forward_cuda(const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& center,
                                 const at::Tensor& gama,
                                 const float spatial_scale,
                                 const int pooled_height) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");
  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto num_guss= center.size(0);

  auto output = at::empty({num_guss ,num_rois, channels, pooled_height}, input.options());
  auto output_size = num_guss * num_rois * pooled_height * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)output_size, 512L), 4096L));
  dim3 block(512);

  // printf("Debug main function: height:%d\n", height);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "Super1D_forward", [&] {
    Super1DForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data<scalar_t>(),
         center.contiguous().data<scalar_t>(),
         gama.contiguous().data<scalar_t>(),
         spatial_scale,
         channels,
         height,
         num_rois,
         pooled_height,
         rois.contiguous().data<scalar_t>(),
         output.data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
std::vector<at::Tensor> Super_backward_cuda(const at::Tensor& grad,
                                  const at::Tensor& input,
                                  const at::Tensor& rois,
                                  const at::Tensor& center,
                                  const at::Tensor& gama,
                                  const float spatial_scale,
                                  const int pooled_height,
                                  const int batch_size,
                                  const int channels,
                                  const int height) {
  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto num_guss= center.size(0);
  auto grad_input = at::zeros({batch_size, channels, height}, grad.options());
  auto grad_center = at::zeros({num_guss}, grad.options());
  auto grad_gama = at::zeros({num_guss}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return {grad_input,grad_center,grad_gama};
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "Super1D_backward", [&] {
    Super1DBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data<scalar_t>(),
         input.contiguous().data<scalar_t>(),
         center.contiguous().data<scalar_t>(),
         gama.contiguous().data<scalar_t>(),
         num_rois,
         spatial_scale,
         channels,
         height,
         pooled_height,
         grad_input.data<scalar_t>(),
         grad_center.data<scalar_t>(),
         grad_gama.data<scalar_t>(),
         rois.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return {grad_input,grad_center,grad_gama};
}
