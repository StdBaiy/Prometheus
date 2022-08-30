#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <vector>
#include <string>
#include "NvInfer.h"

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int INPUT_H = 608;
    static constexpr int INPUT_W = 608;

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection {
        //center_x center_y w h
        float bbox[LOCATIONS];
        float conf;  // bbox_conf * cls_conf
        float class_id;
    };
}

namespace nvinfer1
{
    class YoloLayerPlugin : public IPluginV2IOExt
    {
    public:
        YoloLayerPlugin(int32_t classCount, int32_t netWidth, int32_t netHeight, int32_t maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
        YoloLayerPlugin(const void* data, size_t length);
        ~YoloLayerPlugin();

        int32_t getNbOutputs() const noexcept override
        {
            return 1;
        }

        Dims getOutputDimensions(int32_t index, const Dims* inputs, int32_t nbInputDims) noexcept override;

        int32_t initialize() noexcept override;

        virtual void terminate() noexcept override {};

        virtual size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override { return 0; }

        virtual int32_t enqueue(int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

        virtual size_t getSerializationSize() const noexcept override;

        virtual void serialize(void* buffer) const noexcept override;

        bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) const noexcept override {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char* getPluginType() const noexcept override;

        const char* getPluginVersion() const noexcept override;

        void destroy() noexcept override;

        IPluginV2IOExt* clone() const noexcept override;

        void setPluginNamespace(const char* pluginNamespace) noexcept override;

        const char* getPluginNamespace() const noexcept override;

        DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, int32_t nbInputs) const noexcept override;

        bool isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool* inputIsBroadcasted, int32_t nbInputs) const noexcept override;

        bool canBroadcastInputAcrossBatch(int32_t inputIndex) const noexcept override;

        void attachToContext(
            cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) noexcept override;

        void configurePlugin(const PluginTensorDesc* in, int32_t nbInput, const PluginTensorDesc* out, int32_t nbOutput) noexcept override;

        void detachFromContext() noexcept override;

    private:
        void forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int32_t batchSize = 1);
        int32_t mThreadCount = 256;
        const char* mPluginNamespace;
        int32_t mKernelCount;
        int32_t mClassCount;
        int32_t mYoloV5NetWidth;
        int32_t mYoloV5NetHeight;
        int32_t mMaxOutObject;
        std::vector<Yolo::YoloKernel> mYoloKernel;
        void** mAnchor;
    };

    class YoloPluginCreator : public IPluginCreator
    {
    public:
        YoloPluginCreator();

        ~YoloPluginCreator() override = default;

        const char* getPluginName() const noexcept override;

        const char* getPluginVersion() const noexcept override;

        const PluginFieldCollection* getFieldNames() noexcept override;

        IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

        IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

        void setPluginNamespace(const char* libNamespace) noexcept override
        {
            mNamespace = libNamespace;
        }

        const char* getPluginNamespace() const noexcept override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);
};

#endif 
