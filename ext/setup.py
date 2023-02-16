from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name="GuideConv",
    ext_modules=CUDAExtension(sources=["guideconv.cc", "guideconv.cu"]),
)
