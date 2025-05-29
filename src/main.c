#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Include generated shader headers (from CMake shader compilation)
#include "shader2d_vert_spv.h"
#include "shader2d_frag_spv.h"

// Error handling macro
#define VK_CHECK(call) \
    do { \
        VkResult result_ = (call); \
        if (result_ != VK_SUCCESS) { \
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Vulkan error at %s:%d: %d", __FILE__, __LINE__, result_); \
            return false; \
        } \
    } while (0)

typedef struct {
    SDL_Window *window;
    VkInstance instance;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    uint32_t graphicsQueueFamily;
    VkQueue graphicsQueue;
    VkSwapchainKHR swapchain;
    VkImage *swapchainImages;
    uint32_t swapchainImageCount;
    VkImageView *swapchainImageViews;
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkFramebuffer *framebuffers;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
    VkSemaphore imageAvailableSemaphore;
    VkSemaphore renderFinishedSemaphore;
    VkFence inFlightFence;
    VkBuffer uniformBuffer;              // New: Uniform buffer
    VkDeviceMemory uniformBufferMemory;  // New: Uniform buffer memory
    VkDescriptorSetLayout descriptorSetLayout; // New: Descriptor set layout
    VkDescriptorPool descriptorPool;     // New: Descriptor pool
    VkDescriptorSet descriptorSet;       // New: Descriptor set
    float offsetX;                       // New: Track X offset
    float offsetY;                       // New: Track Y offset
} VulkanContext;


bool init_vulkan(VulkanContext *ctx) {
    // Create Vulkan instance
    uint32_t extensionCount = 0;
    const char *const *extensions = SDL_Vulkan_GetInstanceExtensions(&extensionCount);
    if (!extensions) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to get Vulkan extensions: %s", SDL_GetError());
        return false;
    }

    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Vulkan Triangle",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0
    };

    VkInstanceCreateInfo createInfo = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = extensionCount,
        .ppEnabledExtensionNames = extensions,
        .enabledLayerCount = 0
    };

    VK_CHECK(vkCreateInstance(&createInfo, NULL, &ctx->instance));

    // Create surface
    if (!SDL_Vulkan_CreateSurface(ctx->window, ctx->instance, NULL, &ctx->surface)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to create Vulkan surface: %s", SDL_GetError());
        vkDestroyInstance(ctx->instance, NULL);
        return false;
    }

    // Select physical device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(ctx->instance, &deviceCount, NULL);
    if (deviceCount == 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "No Vulkan physical devices found");
        return false;
    }
    VkPhysicalDevice *devices = malloc(deviceCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx->instance, &deviceCount, devices);
    ctx->physicalDevice = devices[0]; // Pick first device
    free(devices);

    // Find graphics queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physicalDevice, &queueFamilyCount, NULL);
    VkQueueFamilyProperties *queueFamilies = malloc(queueFamilyCount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physicalDevice, &queueFamilyCount, queueFamilies);
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            ctx->graphicsQueueFamily = i;
            break;
        }
    }
    free(queueFamilies);

    // Create logical device
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx->graphicsQueueFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
    };

    VkDeviceCreateInfo deviceCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueCreateInfo,
        .enabledExtensionCount = 1,
        .ppEnabledExtensionNames = (const char *const[]){"VK_KHR_swapchain"},
        .enabledLayerCount = 0
    };

    VK_CHECK(vkCreateDevice(ctx->physicalDevice, &deviceCreateInfo, NULL, &ctx->device));
    vkGetDeviceQueue(ctx->device, ctx->graphicsQueueFamily, 0, &ctx->graphicsQueue);

    return true;
}

bool create_swapchain(VulkanContext *ctx) {
    VkSurfaceCapabilitiesKHR capabilities;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx->physicalDevice, ctx->surface, &capabilities));

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx->physicalDevice, ctx->surface, &formatCount, NULL);
    VkSurfaceFormatKHR *formats = malloc(formatCount * sizeof(VkSurfaceFormatKHR));
    vkGetPhysicalDeviceSurfaceFormatsKHR(ctx->physicalDevice, ctx->surface, &formatCount, formats);
    VkSurfaceFormatKHR format = formats[0]; // Pick first format
    free(formats);

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx->physicalDevice, ctx->surface, &presentModeCount, NULL);
    VkPresentModeKHR *presentModes = malloc(presentModeCount * sizeof(VkPresentModeKHR));
    vkGetPhysicalDeviceSurfacePresentModesKHR(ctx->physicalDevice, ctx->surface, &presentModeCount, presentModes);
    VkPresentModeKHR presentMode = presentModes[0]; // Pick first mode
    free(presentModes);

    VkSwapchainCreateInfoKHR swapchainCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = ctx->surface,
        .minImageCount = capabilities.minImageCount + 1,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = capabilities.currentExtent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = presentMode,
        .clipped = VK_TRUE,
        .oldSwapchain = VK_NULL_HANDLE
    };

    VK_CHECK(vkCreateSwapchainKHR(ctx->device, &swapchainCreateInfo, NULL, &ctx->swapchain));

    vkGetSwapchainImagesKHR(ctx->device, ctx->swapchain, &ctx->swapchainImageCount, NULL);
    ctx->swapchainImages = malloc(ctx->swapchainImageCount * sizeof(VkImage));
    vkGetSwapchainImagesKHR(ctx->device, ctx->swapchain, &ctx->swapchainImageCount, ctx->swapchainImages);

    ctx->swapchainImageViews = malloc(ctx->swapchainImageCount * sizeof(VkImageView));
    for (uint32_t i = 0; i < ctx->swapchainImageCount; i++) {
        VkImageViewCreateInfo viewInfo = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = ctx->swapchainImages[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format.format,
            .components = {VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .subresourceRange.baseMipLevel = 0,
            .subresourceRange.levelCount = 1,
            .subresourceRange.baseArrayLayer = 0,
            .subresourceRange.layerCount = 1
        };
        VK_CHECK(vkCreateImageView(ctx->device, &viewInfo, NULL, &ctx->swapchainImageViews[i]));
    }

    return true;
}

bool create_render_pass(VulkanContext *ctx) {
    VkAttachmentDescription colorAttachment = {
        .format = VK_FORMAT_B8G8R8A8_SRGB,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    };

    VkAttachmentReference colorAttachmentRef = {
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    };

    VkSubpassDescription subpass = {
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachmentRef
    };

    VkRenderPassCreateInfo renderPassInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &colorAttachment,
        .subpassCount = 1,
        .pSubpasses = &subpass
    };

    VK_CHECK(vkCreateRenderPass(ctx->device, &renderPassInfo, NULL, &ctx->renderPass));
    return true;
}

bool create_descriptor_set_layout(VulkanContext *ctx) {
    VkDescriptorSetLayoutBinding uboLayoutBinding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .pImmutableSamplers = NULL
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &uboLayoutBinding
    };

    VK_CHECK(vkCreateDescriptorSetLayout(ctx->device, &layoutInfo, NULL, &ctx->descriptorSetLayout));
    return true;
}

bool create_uniform_buffer(VulkanContext *ctx) {
    VkDeviceSize bufferSize = sizeof(float) * 2; // vec2 offset

    VkBufferCreateInfo bufferInfo = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bufferSize,
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE
    };

    VK_CHECK(vkCreateBuffer(ctx->device, &bufferInfo, NULL, &ctx->uniformBuffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(ctx->device, ctx->uniformBuffer, &memRequirements);

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(ctx->physicalDevice, &memProperties);

    uint32_t memoryTypeIndex = UINT32_MAX;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & 
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) {
            memoryTypeIndex = i;
            break;
        }
    }
    if (memoryTypeIndex == UINT32_MAX) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to find suitable memory type for uniform buffer");
        return false;
    }

    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memRequirements.size,
        .memoryTypeIndex = memoryTypeIndex
    };

    VK_CHECK(vkAllocateMemory(ctx->device, &allocInfo, NULL, &ctx->uniformBufferMemory));
    VK_CHECK(vkBindBufferMemory(ctx->device, ctx->uniformBuffer, ctx->uniformBufferMemory, 0));

    return true;
}

bool create_descriptor_pool_and_set(VulkanContext *ctx) {
    VkDescriptorPoolSize poolSize = {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1
    };

    VkDescriptorPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize
    };

    VK_CHECK(vkCreateDescriptorPool(ctx->device, &poolInfo, NULL, &ctx->descriptorPool));

    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx->descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &ctx->descriptorSetLayout
    };

    VK_CHECK(vkAllocateDescriptorSets(ctx->device, &allocInfo, &ctx->descriptorSet));

    VkDescriptorBufferInfo bufferInfo = {
        .buffer = ctx->uniformBuffer,
        .offset = 0,
        .range = sizeof(float) * 2
    };

    VkWriteDescriptorSet descriptorWrite = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ctx->descriptorSet,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &bufferInfo
    };

    vkUpdateDescriptorSets(ctx->device, 1, &descriptorWrite, 0, NULL);
    return true;
}

bool create_graphics_pipeline(VulkanContext *ctx) {
    VkShaderModule vertShaderModule;
    VkShaderModuleCreateInfo vertShaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(shader2d_vert_spv),
        .pCode = shader2d_vert_spv
    };
    VK_CHECK(vkCreateShaderModule(ctx->device, &vertShaderInfo, NULL, &vertShaderModule));

    VkShaderModule fragShaderModule;
    VkShaderModuleCreateInfo fragShaderInfo = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(shader2d_frag_spv),
        .pCode = shader2d_frag_spv
    };
    VK_CHECK(vkCreateShaderModule(ctx->device, &fragShaderInfo, NULL, &fragShaderModule));

    VkPipelineShaderStageCreateInfo shaderStages[] = {
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = vertShaderModule,
            .pName = "main"
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = fragShaderModule,
            .pName = "main"
        }
    };

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 0,
        .vertexAttributeDescriptionCount = 0
    };

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE
    };

    VkViewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = 800.0f,
        .height = 600.0f,
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };

    VkRect2D scissor = {
        .offset = {0, 0},
        .extent = {800, 600}
    };

    VkPipelineViewportStateCreateInfo viewportState = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor
    };

    VkPipelineRasterizationStateCreateInfo rasterizer = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f
    };

    VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE
    };

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT
    };

    VkPipelineColorBlendStateCreateInfo colorBlending = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &colorBlendAttachment
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,                        // Changed from 0
        .pSetLayouts = &ctx->descriptorSetLayout,   // New
        .pushConstantRangeCount = 0
    };

    VK_CHECK(vkCreatePipelineLayout(ctx->device, &pipelineLayoutInfo, NULL, &ctx->pipelineLayout));

    VkGraphicsPipelineCreateInfo pipelineInfo = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2,
        .pStages = shaderStages,
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &colorBlending,
        .layout = ctx->pipelineLayout,
        .renderPass = ctx->renderPass,
        .subpass = 0
    };

    VK_CHECK(vkCreateGraphicsPipelines(ctx->device, VK_NULL_HANDLE, 1, &pipelineInfo, NULL, &ctx->graphicsPipeline));

    vkDestroyShaderModule(ctx->device, fragShaderModule, NULL);
    vkDestroyShaderModule(ctx->device, vertShaderModule, NULL);
    return true;
}

bool create_framebuffers(VulkanContext *ctx) {
    ctx->framebuffers = malloc(ctx->swapchainImageCount * sizeof(VkFramebuffer));
    for (uint32_t i = 0; i < ctx->swapchainImageCount; i++) {
        VkFramebufferCreateInfo framebufferInfo = {
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = ctx->renderPass,
            .attachmentCount = 1,
            .pAttachments = &ctx->swapchainImageViews[i],
            .width = 800,
            .height = 600,
            .layers = 1
        };
        VK_CHECK(vkCreateFramebuffer(ctx->device, &framebufferInfo, NULL, &ctx->framebuffers[i]));
    }
    return true;
}

bool create_command_buffers(VulkanContext *ctx) {
    VkCommandPoolCreateInfo poolInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx->graphicsQueueFamily,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    };
    VK_CHECK(vkCreateCommandPool(ctx->device, &poolInfo, NULL, &ctx->commandPool));

    VkCommandBufferAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1
    };
    VK_CHECK(vkAllocateCommandBuffers(ctx->device, &allocInfo, &ctx->commandBuffer));
    return true;
}

bool create_sync_objects(VulkanContext *ctx) {
    VkSemaphoreCreateInfo semaphoreInfo = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
    };
    VkFenceCreateInfo fenceInfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT
    };
    VK_CHECK(vkCreateSemaphore(ctx->device, &semaphoreInfo, NULL, &ctx->imageAvailableSemaphore));
    VK_CHECK(vkCreateSemaphore(ctx->device, &semaphoreInfo, NULL, &ctx->renderFinishedSemaphore));
    VK_CHECK(vkCreateFence(ctx->device, &fenceInfo, NULL, &ctx->inFlightFence));
    return true;
}

bool update_uniform_buffer(VulkanContext *ctx) {
    float offset[2] = { ctx->offsetX, ctx->offsetY };
    void *data;
    VK_CHECK(vkMapMemory(ctx->device, ctx->uniformBufferMemory, 0, sizeof(offset), 0, &data));
    memcpy(data, offset, sizeof(offset));
    vkUnmapMemory(ctx->device, ctx->uniformBufferMemory);
    return true;
}

bool record_command_buffer(VulkanContext *ctx, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
    };
    VK_CHECK(vkBeginCommandBuffer(ctx->commandBuffer, &beginInfo));

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    VkRenderPassBeginInfo renderPassInfo = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = ctx->renderPass,
        .framebuffer = ctx->framebuffers[imageIndex],
        .renderArea.offset = {0, 0},
        .renderArea.extent = {800, 600},
        .clearValueCount = 1,
        .pClearValues = &clearColor
    };

    vkCmdBeginRenderPass(ctx->commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(ctx->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->graphicsPipeline);
    vkCmdBindDescriptorSets(ctx->commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, ctx->pipelineLayout, 0, 1, &ctx->descriptorSet, 0, NULL);
    vkCmdDraw(ctx->commandBuffer, 3, 1, 0, 0);
    vkCmdEndRenderPass(ctx->commandBuffer);
    VK_CHECK(vkEndCommandBuffer(ctx->commandBuffer));
    return true;
}

void cleanup(VulkanContext *ctx) {
    vkDeviceWaitIdle(ctx->device);
    vkDestroySemaphore(ctx->device, ctx->renderFinishedSemaphore, NULL);
    vkDestroySemaphore(ctx->device, ctx->imageAvailableSemaphore, NULL);
    vkDestroyFence(ctx->device, ctx->inFlightFence, NULL);
    vkDestroyCommandPool(ctx->device, ctx->commandPool, NULL);
    for (uint32_t i = 0; i < ctx->swapchainImageCount; i++) {
        vkDestroyFramebuffer(ctx->device, ctx->framebuffers[i], NULL);
        vkDestroyImageView(ctx->device, ctx->swapchainImageViews[i], NULL);
    }
    free(ctx->framebuffers);
    free(ctx->swapchainImages);
    free(ctx->swapchainImageViews);
    vkDestroyPipeline(ctx->device, ctx->graphicsPipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, ctx->pipelineLayout, NULL);
    vkDestroyRenderPass(ctx->device, ctx->renderPass, NULL);
    vkDestroySwapchainKHR(ctx->device, ctx->swapchain, NULL);
    vkDestroyDescriptorPool(ctx->device, ctx->descriptorPool, NULL);        // New
    vkDestroyDescriptorSetLayout(ctx->device, ctx->descriptorSetLayout, NULL); // New
    vkDestroyBuffer(ctx->device, ctx->uniformBuffer, NULL);                 // New
    vkFreeMemory(ctx->device, ctx->uniformBufferMemory, NULL);             // New
    vkDestroyDevice(ctx->device, NULL);
    vkDestroySurfaceKHR(ctx->instance, ctx->surface, NULL);
    vkDestroyInstance(ctx->instance, NULL);
    SDL_DestroyWindow(ctx->window);
    SDL_Quit();
}

int main(int argc, char *argv[]) {
    VulkanContext *ctx = calloc(1, sizeof(VulkanContext)); // Allocate and zero-initialize
    if (!ctx) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to allocate VulkanContext");
        return 1;
    }

    ctx->window = SDL_CreateWindow("Vulkan Triangle", 800, 600, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);
    if (!ctx->window) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to create window: %s", SDL_GetError());
        free(ctx);
        SDL_Quit();
        return 1;
    }

    if (!init_vulkan(ctx) ||
        !create_swapchain(ctx) ||
        !create_descriptor_set_layout(ctx) ||  // New
        !create_uniform_buffer(ctx) ||        // New
        !create_descriptor_pool_and_set(ctx) || // New
        !create_render_pass(ctx) ||
        !create_graphics_pipeline(ctx) ||
        !create_framebuffers(ctx) ||
        !create_command_buffers(ctx) ||
        !create_sync_objects(ctx)) {
        cleanup(ctx);
        free(ctx);
        return 1;
    }

    
    bool running = true;
    SDL_Event event;
    float moveSpeed = 0.01f; // Adjust as needed
    bool keyState[4] = { false, false, false, false }; // W, A, S, D

    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (event.type == SDL_EVENT_KEY_DOWN) {
                switch (event.key.key) {
                    case SDLK_W: keyState[0] = true; break;
                    case SDLK_A: keyState[1] = true; break;
                    case SDLK_S: keyState[2] = true; break;
                    case SDLK_D: keyState[3] = true; break;
                }
            } else if (event.type == SDL_EVENT_KEY_UP) {
                switch (event.key.key) {
                    case SDLK_W: keyState[0] = false; break;
                    case SDLK_A: keyState[1] = false; break;
                    case SDLK_S: keyState[2] = false; break;
                    case SDLK_D: keyState[3] = false; break;
                }
            }
        }

        // Update offsets based on key states
        if (keyState[0]) ctx->offsetY -= moveSpeed; // W: Move up
        if (keyState[1]) ctx->offsetX -= moveSpeed; // A: Move left
        if (keyState[2]) ctx->offsetY += moveSpeed; // S: Move down
        if (keyState[3]) ctx->offsetX += moveSpeed; // D: Move right
        ctx->offsetX = SDL_clamp(ctx->offsetX, -1.0f, 1.0f);
        ctx->offsetY = SDL_clamp(ctx->offsetY, -1.0f, 1.0f);

        if (!update_uniform_buffer(ctx)) {
            cleanup(ctx);
            free(ctx);
            return 1;
        }

        vkWaitForFences(ctx->device, 1, &ctx->inFlightFence, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx->device, 1, &ctx->inFlightFence);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(ctx->device, ctx->swapchain, UINT64_MAX, ctx->imageAvailableSemaphore, VK_NULL_HANDLE, &imageIndex);
        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            // Handle swapchain recreation if needed
            continue;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to acquire swapchain image: %d", result);
            cleanup(ctx);
            free(ctx);
            return 1;
        }

        vkResetCommandBuffer(ctx->commandBuffer, 0);
        if (!record_command_buffer(ctx, imageIndex)) {
            cleanup(ctx);
            free(ctx);
            return 1;
        }

        VkSubmitInfo submitInfo = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &ctx->imageAvailableSemaphore,
            .pWaitDstStageMask = (VkPipelineStageFlags[]){VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
            .commandBufferCount = 1,
            .pCommandBuffers = &ctx->commandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &ctx->renderFinishedSemaphore
        };

        VK_CHECK(vkQueueSubmit(ctx->graphicsQueue, 1, &submitInfo, ctx->inFlightFence));

        VkPresentInfoKHR presentInfo = {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &ctx->renderFinishedSemaphore,
            .swapchainCount = 1,
            .pSwapchains = &ctx->swapchain,
            .pImageIndices = &imageIndex
        };

        result = vkQueuePresentKHR(ctx->graphicsQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            // Handle swapchain recreation if needed
        } else if (result != VK_SUCCESS) {
            SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Failed to present: %d", result);
            cleanup(ctx);
            free(ctx);
            return 1;
        }
    }


    cleanup(ctx);
    free(ctx);
    return 0;
}


//