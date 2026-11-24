// GPU side of the host: owns the D3D11 device, the swap textures handed to the
// SteamVR compositor, the lens-distortion pass, RGB->NV12 conversion and the
// hardware H.264 encoder (Media Foundation).
#pragma once
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <mfidl.h>
#include <mftransform.h>
#include <wrl/client.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common.h"

namespace phonevr {

using Microsoft::WRL::ComPtr;

class H264Encoder;

class VideoPipeline {
public:
    using PacketFn = std::function<void(const uint8_t* data, size_t size, bool keyframe,
                                        uint64_t frameId, uint64_t hostTimeUs)>;

    VideoPipeline();
    ~VideoPipeline();

    bool Init(const Settings& s);
    void Shutdown();

    uint64_t AdapterLuid() const { return luid_; }
    void SetPacketCallback(PacketFn fn) { onPacket_ = std::move(fn); }

    void SetBitrate(uint32_t kbps);
    uint32_t Bitrate() const { return bitrateKbps_; }
    void RequestKeyframe();

    // ---- direct-mode plumbing (called from the SteamVR compositor) ----
    void CreateSwapTextureSet(uint32_t pid, uint32_t w, uint32_t h, uint32_t dxgiFormat,
                              uint32_t samples, uint64_t outHandles[3]);
    void DestroySwapTextureSet(uint64_t handle);
    void DestroyAllSwapTextureSets(uint32_t pid);
    void NextSwapTextureSetIndex(const uint64_t handles[2], uint32_t indices[2]);
    void SubmitLayer(const uint64_t handles[2], const float bounds[2][4]);
    void Present(uint64_t syncTextureHandle);

private:
    struct SwapSet;
    struct SwapTex {
        ComPtr<ID3D11Texture2D> tex;
        ComPtr<ID3D11ShaderResourceView> srv;
        bool encodeSrgb = false;
        std::shared_ptr<SwapSet> set;
        int index = 0;
    };
    struct SwapSet {
        uint32_t pid = 0;
        uint64_t handles[3] = {};
    };
    struct Slot {
        ComPtr<ID3D11Texture2D> tex;
        ComPtr<ID3D11VideoProcessorOutputView> view;
        bool busy = false;
    };

    bool CreateDevice(const Settings& s);
    bool CreateRenderResources();
    bool CreateConverter();
    void EncodeLoop();
    // One compositor layer (scene, overlay, ...) as handed over by SubmitLayer.
    struct Layer {
        uint64_t handles[2] = {};
        float bounds[2][4] = {};
    };

    void DrawEye(int eye, const SwapTex* src, const float bounds[4], bool opaque);

    Settings settings_;
    uint64_t luid_ = 0;
    PacketFn onPacket_;

    ComPtr<ID3D11Device> dev_;
    ComPtr<ID3D11Device1> dev1_;
    ComPtr<ID3D11DeviceContext> ctx_;
    ComPtr<ID3D11VideoDevice> vdev_;
    ComPtr<ID3D11VideoContext> vctx_;

    // distortion pass
    ComPtr<ID3D11VertexShader> vs_;
    ComPtr<ID3D11PixelShader> ps_;
    ComPtr<ID3D11SamplerState> sampler_;
    ComPtr<ID3D11Buffer> cbuf_;
    ComPtr<ID3D11BlendState> blendOver_;  // alpha-blends overlay layers onto the scene
    ComPtr<ID3D11Texture2D> rgbTex_;
    ComPtr<ID3D11RenderTargetView> rgbRtv_;

    // converter
    ComPtr<ID3D11VideoProcessorEnumerator> vpEnum_;
    ComPtr<ID3D11VideoProcessor> vp_;
    ComPtr<ID3D11VideoProcessorInputView> vpIn_;
    static constexpr int kSlots = 4;
    Slot slots_[kSlots];

    // swap texture registry
    std::mutex texMutex_;
    std::map<uint64_t, SwapTex> swapTex_;
    std::map<uint64_t, ComPtr<ID3D11Texture2D>> syncTex_;
    // The compositor calls SubmitLayer once per layer (bottom to top), then Present.
    std::vector<Layer> frameLayers_;  // layers submitted since the last Present
    std::vector<Layer> lastLayers_;   // reused if a Present arrives with no new layers
    size_t loggedLayerCount_ = 0;

    // encoder hand-off
    std::unique_ptr<H264Encoder> enc_;
    std::thread encThread_;
    std::mutex frameMutex_;
    std::condition_variable frameCv_;
    int pendingSlot_ = -1;
    bool stop_ = false;
    std::atomic<uint32_t> bitrateKbps_{30000};
    std::atomic<uint64_t> frameCounter_{0};
};

}  // namespace phonevr
