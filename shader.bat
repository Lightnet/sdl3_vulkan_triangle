@echo off
setlocal

set VULKAN_VERSION=1.4.313.0
set VULKAN_Path=C:\VulkanSDK\%VULKAN_VERSION%\Bin\glslangValidator.exe
echo path "%VULKAN_Path%"

%VULKAN_Path% -V --vn shader2d_vert_spv shaders/shader2d.vert -o include/shader2d_vert_spv.h
%VULKAN_Path% -V --vn shader2d_frag_spv shaders/shader2d.frag -o include/shader2d_frag_spv.h

endlocal