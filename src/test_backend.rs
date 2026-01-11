#[cfg(all(test, feature = "backend-wgpu"))]
pub type TestBackend = burn::backend::Wgpu;

#[cfg(all(test, feature = "backend-wgpu"))]
pub fn test_device() -> <TestBackend as burn::tensor::backend::Backend>::Device {
    burn::backend::wgpu::WgpuDevice::default()
}

#[cfg(all(test, feature = "backend-tch"))]
pub type TestBackend = burn::backend::LibTorch<f32>;

#[cfg(all(test, feature = "backend-tch"))]
pub fn test_device() -> <TestBackend as burn::tensor::backend::Backend>::Device {
    burn::backend::libtorch::LibTorchDevice::Cpu
}

#[cfg(all(test, feature = "backend-cuda"))]
pub type TestBackend = burn_cuda::Cuda;

#[cfg(all(test, feature = "backend-cuda"))]
pub fn test_device() -> <TestBackend as burn::tensor::backend::Backend>::Device {
    burn_cuda::CudaDevice::default()
}

#[cfg(all(
    test,
    not(feature = "backend-wgpu"),
    not(feature = "backend-tch"),
    not(feature = "backend-cuda")
))]
compile_error!("At least one backend feature must be enabled for tests: backend-wgpu, backend-tch, or backend-cuda");
