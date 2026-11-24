#include "video_pipeline.h"

#include <codecapi.h>
#include <d3d11_4.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <mfapi.h>
#include <mferror.h>
#include <oleauto.h>
#include <strmif.h>

#include <chrono>
#include <cmath>
#include <cstring>

namespace phonevr {

namespace {

uint64_t NowUs() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// Lens pre-distortion shader. One full-screen triangle per eye viewport.
//
//  uv (0..1 over the eye viewport) -> lens space v (units of half-height,
//  origin at the lens centre) -> radial barrel factor f(r) -> source texture uv.
//  Because the rendered frustum is symmetric and measured in the same units,
//  the source uv is simply 0.5 + v*f/scale.
// ---------------------------------------------------------------------------
const char* kShader = R"(
cbuffer P : register(b0) {
    float2 lensCenter;
    float2 scale;      // (2*aspect, 2)
    float4 bounds;     // source texture valid region: uMin, vMin, uMax, vMax
    float  k1; float k2; float chroma; float encode;
    float  opaque;     // 1 = bottom (scene) layer: ignore alpha; 0 = overlay layer: keep alpha
    float3 pad;
};
Texture2D tex : register(t0);
SamplerState smp : register(s0);

struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

VSOut VS(uint id : SV_VertexID) {
    VSOut o;
    float2 t = float2((id << 1) & 2, id & 2);
    o.uv = t;
    o.pos = float4(t * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}

float3 srgbEncode(float3 c) {
    c = saturate(c);
    float3 lo = c * 12.92;
    float3 hi = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    return lerp(lo, hi, step(0.0031308, c));
}

float4 PS(VSOut i) : SV_Target {
    float2 v = (i.uv - lensCenter) * scale;
    float r2 = dot(v, v);
    float f = 1.0 + k1 * r2 + k2 * r2 * r2;
    float3 rgb = float3(0, 0, 0);
    float a = 0.0;  // outside the texture: black for the scene, transparent for overlays
    [unroll] for (int ch = 0; ch < 3; ch++) {
        float cf = f * (1.0 + chroma * (ch - 1));   // R: -chroma, G: 0, B: +chroma
        float2 uv = 0.5 + (v * cf) / scale;
        if (all(uv >= 0.0) && all(uv <= 1.0)) {
            float2 tuv = lerp(bounds.xy, bounds.zw, uv);
            float4 s = tex.SampleLevel(smp, tuv, 0);
            rgb[ch] = s[ch];
            if (ch == 1) a = s.a;
        }
    }
    if (encode > 0.5) rgb = srgbEncode(rgb);
    // The scene's alpha is meaningless (often 0), so the bottom layer is always opaque.
    return float4(rgb, opaque > 0.5 ? 1.0 : saturate(a));
}
)";

struct CBData {
    float lensCenter[2];
    float scale[2];
    float bounds[4];
    float k1, k2, chroma, encode;
    float opaque, pad[3];
};
static_assert(sizeof(CBData) == 64, "cbuffer layout");

DXGI_FORMAT SrvFormatFor(DXGI_FORMAT f, bool& encodeSrgb) {
    encodeSrgb = false;
    switch (f) {
        case DXGI_FORMAT_R8G8B8A8_TYPELESS:
            encodeSrgb = true;
            return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        case DXGI_FORMAT_B8G8R8A8_TYPELESS:
            encodeSrgb = true;
            return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
            encodeSrgb = true;  // sampler returns linear, the stream wants sRGB-encoded
            return f;
        case DXGI_FORMAT_R16G16B16A16_TYPELESS:
            encodeSrgb = true;
            return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
            encodeSrgb = true;
            return f;
        case DXGI_FORMAT_R10G10B10A2_TYPELESS:
            return DXGI_FORMAT_R10G10B10A2_UNORM;
        default:
            return f;
    }
}

// ---------------------------------------------------------------------------
// H.264 encoder on top of a hardware Media Foundation transform.
// ---------------------------------------------------------------------------
HRESULT SetU32(ICodecAPI* c, const GUID& key, uint32_t v) {
    VARIANT var;
    VariantInit(&var);
    var.vt = VT_UI4;
    var.ulVal = v;
    return c->SetValue(&key, &var);
}
HRESULT SetBool(ICodecAPI* c, const GUID& key, bool v) {
    VARIANT var;
    VariantInit(&var);
    var.vt = VT_BOOL;
    var.boolVal = v ? VARIANT_TRUE : VARIANT_FALSE;
    return c->SetValue(&key, &var);
}

// Scan an Annex-B buffer for a NAL unit type (only the first few NALs matter).
bool HasNal(const uint8_t* d, size_t n, int type) {
    for (size_t i = 0; i + 3 < n && i < 4096; i++) {
        if (d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1) {
            if ((d[i + 3] & 0x1f) == type) return true;
            i += 2;
        }
    }
    return false;
}

}  // namespace

class H264Encoder {
public:
    using OutFn = std::function<void(const uint8_t*, size_t, bool)>;

    bool Init(ID3D11Device* dev, uint32_t w, uint32_t h, float fps, uint32_t kbps, bool baseline,
              OutFn out) {
        out_ = std::move(out);
        baseline_ = baseline;
        w_ = w;
        h_ = h;
        fps_ = fps;
        frameDurationHns_ = (LONGLONG)(1e7 / fps);

        HRESULT hr = MFStartup(MF_VERSION, MFSTARTUP_FULL);
        if (FAILED(hr)) return false;
        mfStarted_ = true;

        UINT32 token = 0;
        hr = MFCreateDXGIDeviceManager(&token, &mgr_);
        if (FAILED(hr) || FAILED(mgr_->ResetDevice(dev, token))) return false;

        MFT_REGISTER_TYPE_INFO inInfo = {MFMediaType_Video, MFVideoFormat_NV12};
        MFT_REGISTER_TYPE_INFO outInfo = {MFMediaType_Video, MFVideoFormat_H264};
        IMFActivate** acts = nullptr;
        UINT32 count = 0;
        hr = MFTEnumEx(MFT_CATEGORY_VIDEO_ENCODER, MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_SORTANDFILTER,
                       &inInfo, &outInfo, &acts, &count);
        if (FAILED(hr) || count == 0) {
            Log("PhoneVR: no hardware H.264 encoder found (Media Foundation)");
            return false;
        }
        bool ok = false;
        for (UINT32 i = 0; i < count; i++) {
            if (!ok) {
                mft_.Reset();
                if (SUCCEEDED(acts[i]->ActivateObject(IID_PPV_ARGS(&mft_))) &&
                    Configure(token, kbps)) {
                    ok = true;
                } else {
                    mft_.Reset();
                }
            }
            acts[i]->Release();
        }
        CoTaskMemFree(acts);
        if (!ok) {
            Log("PhoneVR: could not configure any hardware H.264 encoder");
            return false;
        }
        return true;
    }

    void Shutdown() {
        if (mft_) {
            mft_->ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0);
            mft_->ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0);
            mft_->ProcessMessage(MFT_MESSAGE_SET_D3D_MANAGER, 0);
        }
        evgen_.Reset();
        codec_.Reset();
        mft_.Reset();
        mgr_.Reset();
        if (mfStarted_) MFShutdown();
        mfStarted_ = false;
    }

    void SetBitrate(uint32_t kbps) { pendingKbps_ = kbps; }
    void ForceKeyframe() { forceIdr_ = true; }

    bool Encode(ID3D11Texture2D* nv12, uint64_t ptsUs) {
        if (!mft_) return false;
        uint32_t kb = pendingKbps_.exchange(0);
        if (kb && codec_) SetU32(codec_.Get(), CODECAPI_AVEncCommonMeanBitRate, kb * 1000);
        if (forceIdr_.exchange(false) && codec_) SetU32(codec_.Get(), CODECAPI_AVEncVideoForceKeyFrame, 1);

        ComPtr<IMFMediaBuffer> buf;
        if (FAILED(MFCreateDXGISurfaceBuffer(__uuidof(ID3D11Texture2D), nv12, 0, FALSE, &buf)))
            return false;
        buf->SetCurrentLength(w_ * h_ * 3 / 2);
        ComPtr<IMFSample> sample;
        if (FAILED(MFCreateSample(&sample))) return false;
        sample->AddBuffer(buf.Get());
        sample->SetSampleTime((LONGLONG)ptsUs * 10);
        sample->SetSampleDuration(frameDurationHns_);

        if (!async_) {
            HRESULT hr = mft_->ProcessInput(0, sample.Get(), 0);
            if (FAILED(hr)) return false;
            while (ReadOutput()) {}
            return true;
        }

        // Asynchronous (typical for hardware) MFT: input is only allowed after
        // METransformNeedInput, output is only readable after METransformHaveOutput.
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
        while (needInput_ == 0) {
            if (!PumpOne(deadline)) return false;
        }
        if (FAILED(mft_->ProcessInput(0, sample.Get(), 0))) return false;
        needInput_--;
        gotOutput_ = false;
        deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
        while (!gotOutput_) {
            if (!PumpOne(deadline)) break;
        }
        return true;
    }

private:
    bool Configure(UINT32 token, uint32_t kbps) {
        (void)token;
        ComPtr<IMFAttributes> attrs;
        if (SUCCEEDED(mft_->GetAttributes(&attrs)) && attrs) {
            UINT32 a = 0;
            if (SUCCEEDED(attrs->GetUINT32(MF_TRANSFORM_ASYNC, &a)) && a) {
                async_ = true;
                attrs->SetUINT32(MF_TRANSFORM_ASYNC_UNLOCK, TRUE);
            }
            attrs->SetUINT32(MF_LOW_LATENCY, TRUE);
        }
        if (FAILED(mft_->ProcessMessage(MFT_MESSAGE_SET_D3D_MANAGER, (ULONG_PTR)mgr_.Get())))
            return false;

        // codec tuning for lowest latency
        if (SUCCEEDED(mft_->QueryInterface(IID_ICodecAPI, reinterpret_cast<void**>(codec_.GetAddressOf())))) {
            SetBool(codec_.Get(), CODECAPI_AVLowLatencyMode, true);
            SetU32(codec_.Get(), CODECAPI_AVEncCommonRateControlMode, eAVEncCommonRateControlMode_CBR);
            SetU32(codec_.Get(), CODECAPI_AVEncCommonMeanBitRate, kbps * 1000);
            SetU32(codec_.Get(), CODECAPI_AVEncMPVGOPSize, 3600);  // keyframes on demand
            SetU32(codec_.Get(), CODECAPI_AVEncMPVDefaultBPictureCount, 0);
            SetU32(codec_.Get(), CODECAPI_AVEncCommonQualityVsSpeed, 30);
            SetU32(codec_.Get(), CODECAPI_AVEncVideoMaxNumRefFrame, 1);  // small decoder DPB = less buffering
        }

        ComPtr<IMFMediaType> out;
        MFCreateMediaType(&out);
        out->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        out->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264);
        out->SetUINT32(MF_MT_AVG_BITRATE, kbps * 1000);
        MFSetAttributeSize(out.Get(), MF_MT_FRAME_SIZE, w_, h_);
        MFSetAttributeRatio(out.Get(), MF_MT_FRAME_RATE, (UINT32)std::lround(fps_), 1);
        MFSetAttributeRatio(out.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
        out->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        out->SetUINT32(MF_MT_MPEG2_PROFILE, baseline_ ? eAVEncH264VProfile_Base : eAVEncH264VProfile_Main);
        out->SetUINT32(MF_MT_VIDEO_NOMINAL_RANGE, MFNominalRange_16_235);
        out->SetUINT32(MF_MT_YUV_MATRIX, MFVideoTransferMatrix_BT709);
        if (FAILED(mft_->SetOutputType(0, out.Get(), 0))) return false;

        ComPtr<IMFMediaType> in;
        MFCreateMediaType(&in);
        in->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        in->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
        MFSetAttributeSize(in.Get(), MF_MT_FRAME_SIZE, w_, h_);
        MFSetAttributeRatio(in.Get(), MF_MT_FRAME_RATE, (UINT32)std::lround(fps_), 1);
        MFSetAttributeRatio(in.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
        in->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
        in->SetUINT32(MF_MT_VIDEO_NOMINAL_RANGE, MFNominalRange_16_235);
        in->SetUINT32(MF_MT_YUV_MATRIX, MFVideoTransferMatrix_BT709);
        if (FAILED(mft_->SetInputType(0, in.Get(), 0))) return false;

        MFT_OUTPUT_STREAM_INFO info = {};
        if (SUCCEEDED(mft_->GetOutputStreamInfo(0, &info))) {
            providesSamples_ = (info.dwFlags & (MFT_OUTPUT_STREAM_PROVIDES_SAMPLES |
                                                MFT_OUTPUT_STREAM_CAN_PROVIDE_SAMPLES)) != 0;
            outSize_ = info.cbSize ? info.cbSize : (1u << 20);
        }
        if (async_ && FAILED(mft_.As(&evgen_))) return false;

        mft_->ProcessMessage(MFT_MESSAGE_COMMAND_FLUSH, 0);
        mft_->ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0);
        mft_->ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0);
        LoadSequenceHeader();
        forceIdr_ = true;
        Log("PhoneVR: H.264 hardware encoder ready (%s MFT, %ux%u @%.0f, %u kbps)",
            async_ ? "async" : "sync", w_, h_, fps_, kbps);
        return true;
    }

    void LoadSequenceHeader() {
        ComPtr<IMFMediaType> t;
        if (FAILED(mft_->GetOutputCurrentType(0, &t))) return;
        UINT32 sz = 0;
        if (SUCCEEDED(t->GetBlobSize(MF_MT_MPEG_SEQUENCE_HEADER, &sz)) && sz > 0) {
            seqHeader_.resize(sz);
            if (FAILED(t->GetBlob(MF_MT_MPEG_SEQUENCE_HEADER, seqHeader_.data(), sz, &sz)))
                seqHeader_.clear();
        }
    }

    // Wait up to `deadline` for one event. Polling (not blocking) so a misbehaving
    // MFT can never hang the encoder thread forever.
    bool PumpOne(std::chrono::steady_clock::time_point deadline) {
        int spins = 0;
        for (;;) {
            ComPtr<IMFMediaEvent> ev;
            HRESULT hr = evgen_->GetEvent(MF_EVENT_FLAG_NO_WAIT, &ev);
            if (SUCCEEDED(hr) && ev) {
                MediaEventType type = MEUnknown;
                ev->GetType(&type);
                if (type == METransformNeedInput) {
                    needInput_++;
                } else if (type == METransformHaveOutput) {
                    ReadOutput();
                    gotOutput_ = true;
                }
                return true;
            }
            if (hr != MF_E_NO_EVENTS_AVAILABLE) return false;
            if (std::chrono::steady_clock::now() >= deadline) return false;
            if (++spins < 50) Sleep(0); else Sleep(1);
        }
    }

    bool ReadOutput() {
        MFT_OUTPUT_DATA_BUFFER out = {};
        ComPtr<IMFSample> own;
        if (!providesSamples_) {
            MFCreateSample(&own);
            ComPtr<IMFMediaBuffer> mb;
            MFCreateMemoryBuffer(outSize_, &mb);
            own->AddBuffer(mb.Get());
            out.pSample = own.Get();
        }
        DWORD status = 0;
        HRESULT hr = mft_->ProcessOutput(0, 1, &out, &status);
        if (hr == MF_E_TRANSFORM_STREAM_CHANGE) {
            ComPtr<IMFMediaType> t;
            if (SUCCEEDED(mft_->GetOutputAvailableType(0, 0, &t))) mft_->SetOutputType(0, t.Get(), 0);
            LoadSequenceHeader();
            return false;
        }
        if (FAILED(hr)) return false;

        IMFSample* s = out.pSample;
        bool ok = false;
        if (s) {
            ComPtr<IMFMediaBuffer> mb;
            if (SUCCEEDED(s->ConvertToContiguousBuffer(&mb))) {
                BYTE* p = nullptr;
                DWORD len = 0, maxLen = 0;
                if (SUCCEEDED(mb->Lock(&p, &maxLen, &len)) && len > 0) {
                    bool key = HasNal(p, len, 5);
                    if (key && !HasNal(p, len, 7)) {
                        if (seqHeader_.empty()) LoadSequenceHeader();
                        std::vector<uint8_t> joined(seqHeader_);
                        joined.insert(joined.end(), p, p + len);
                        if (out_) out_(joined.data(), joined.size(), true);
                    } else if (out_) {
                        out_(p, len, key);
                    }
                    ok = true;
                    mb->Unlock();
                }
            }
            if (providesSamples_) s->Release();
        }
        if (out.pEvents) out.pEvents->Release();
        return ok;
    }

    OutFn out_;
    uint32_t w_ = 0, h_ = 0;
    float fps_ = 60;
    LONGLONG frameDurationHns_ = 166666;
    bool mfStarted_ = false, async_ = false, providesSamples_ = false, baseline_ = true;
    DWORD outSize_ = 1u << 20;
    int needInput_ = 0;
    bool gotOutput_ = false;
    std::atomic<uint32_t> pendingKbps_{0};
    std::atomic<bool> forceIdr_{false};
    std::vector<uint8_t> seqHeader_;
    ComPtr<IMFDXGIDeviceManager> mgr_;
    ComPtr<IMFTransform> mft_;
    ComPtr<IMFMediaEventGenerator> evgen_;
    ComPtr<ICodecAPI> codec_;
};

// ===========================================================================
// VideoPipeline
// ===========================================================================

VideoPipeline::VideoPipeline() = default;
VideoPipeline::~VideoPipeline() { Shutdown(); }

bool VideoPipeline::CreateDevice(const Settings& s) {
    ComPtr<IDXGIFactory1> fac;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&fac)))) return false;
    ComPtr<IDXGIAdapter1> best;
    SIZE_T bestMem = 0;
    for (UINT i = 0;; i++) {
        ComPtr<IDXGIAdapter1> a;
        if (fac->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 d;
        a->GetDesc1(&d);
        if (s.gpuIndex >= 0) {
            if ((int)i == s.gpuIndex) { best = a; break; }
            continue;
        }
        if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        if (!best || d.DedicatedVideoMemory > bestMem) {
            best = a;
            bestMem = d.DedicatedVideoMemory;
        }
    }
    if (!best) return false;
    DXGI_ADAPTER_DESC1 d;
    best->GetDesc1(&d);
    luid_ = ((uint64_t)d.AdapterLuid.HighPart << 32) | d.AdapterLuid.LowPart;
    char name[256];
    WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, name, sizeof(name), nullptr, nullptr);
    Log("PhoneVR: using GPU \"%s\"", name);

    D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    D3D_FEATURE_LEVEL got;
    HRESULT hr = D3D11CreateDevice(best.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                                   D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
                                   levels, 2, D3D11_SDK_VERSION, &dev_, &got, &ctx_);
    if (FAILED(hr)) return false;
    ComPtr<ID3D11Multithread> mt;
    if (SUCCEEDED(ctx_.As(&mt))) mt->SetMultithreadProtected(TRUE);
    dev_.As(&dev1_);
    if (FAILED(dev_.As(&vdev_)) || FAILED(ctx_.As(&vctx_))) {
        Log("PhoneVR: GPU does not expose the D3D11 video processor");
        return false;
    }
    return true;
}

bool VideoPipeline::CreateRenderResources() {
    ComPtr<ID3DBlob> vsb, psb, err;
    UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;
    if (FAILED(D3DCompile(kShader, strlen(kShader), "phonevr", nullptr, nullptr, "VS", "vs_4_0", flags, 0,
                          &vsb, &err)) ||
        FAILED(D3DCompile(kShader, strlen(kShader), "phonevr", nullptr, nullptr, "PS", "ps_4_0", flags, 0,
                          &psb, &err))) {
        Log("PhoneVR: shader compile failed: %s", err ? (const char*)err->GetBufferPointer() : "?");
        return false;
    }
    if (FAILED(dev_->CreateVertexShader(vsb->GetBufferPointer(), vsb->GetBufferSize(), nullptr, &vs_)) ||
        FAILED(dev_->CreatePixelShader(psb->GetBufferPointer(), psb->GetBufferSize(), nullptr, &ps_)))
        return false;

    D3D11_SAMPLER_DESC sd = {};
    sd.Filter = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD = D3D11_FLOAT32_MAX;
    if (FAILED(dev_->CreateSamplerState(&sd, &sampler_))) return false;

    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth = sizeof(CBData);
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(dev_->CreateBuffer(&bd, nullptr, &cbuf_))) return false;

    // Overlay layers (HUD, fade panel, dashboard...) are drawn over the scene with
    // straight alpha; the destination alpha stays 1.
    D3D11_BLEND_DESC bld = {};
    auto& rt = bld.RenderTarget[0];
    rt.BlendEnable = TRUE;
    rt.SrcBlend = D3D11_BLEND_SRC_ALPHA;
    rt.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    rt.BlendOp = D3D11_BLEND_OP_ADD;
    rt.SrcBlendAlpha = D3D11_BLEND_ZERO;
    rt.DestBlendAlpha = D3D11_BLEND_ONE;
    rt.BlendOpAlpha = D3D11_BLEND_OP_ADD;
    rt.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(dev_->CreateBlendState(&bld, &blendOver_))) return false;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = settings_.streamWidth;
    td.Height = settings_.streamHeight;
    td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (FAILED(dev_->CreateTexture2D(&td, nullptr, &rgbTex_)) ||
        FAILED(dev_->CreateRenderTargetView(rgbTex_.Get(), nullptr, &rgbRtv_)))
        return false;
    return true;
}

bool VideoPipeline::CreateConverter() {
    D3D11_VIDEO_PROCESSOR_CONTENT_DESC cd = {};
    cd.InputFrameFormat = D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE;
    cd.InputFrameRate = {(UINT)std::lround(settings_.refreshRate), 1};
    cd.InputWidth = cd.OutputWidth = settings_.streamWidth;
    cd.InputHeight = cd.OutputHeight = settings_.streamHeight;
    cd.OutputFrameRate = cd.InputFrameRate;
    cd.Usage = D3D11_VIDEO_USAGE_OPTIMAL_SPEED;
    if (FAILED(vdev_->CreateVideoProcessorEnumerator(&cd, &vpEnum_)) ||
        FAILED(vdev_->CreateVideoProcessor(vpEnum_.Get(), 0, &vp_)))
        return false;

    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC iv = {};
    iv.ViewDimension = D3D11_VPIV_DIMENSION_TEXTURE2D;
    if (FAILED(vdev_->CreateVideoProcessorInputView(rgbTex_.Get(), vpEnum_.Get(), &iv, &vpIn_)))
        return false;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width = settings_.streamWidth;
    td.Height = settings_.streamHeight;
    td.MipLevels = td.ArraySize = 1;
    td.Format = DXGI_FORMAT_NV12;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    for (auto& s : slots_) {
        D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC ov = {};
        ov.ViewDimension = D3D11_VPOV_DIMENSION_TEXTURE2D;
        if (FAILED(dev_->CreateTexture2D(&td, nullptr, &s.tex)) ||
            FAILED(vdev_->CreateVideoProcessorOutputView(s.tex.Get(), vpEnum_.Get(), &ov, &s.view)))
            return false;
    }

    // Full-range sRGB-ish RGB in, BT.709 limited-range YCbCr out (what the
    // encoder is told in its media type).
    ComPtr<ID3D11VideoContext1> vc1;
    if (SUCCEEDED(vctx_.As(&vc1))) {
        vc1->VideoProcessorSetStreamColorSpace1(vp_.Get(), 0, DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709);
        vc1->VideoProcessorSetOutputColorSpace1(vp_.Get(), DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709);
    }
    return true;
}

bool VideoPipeline::Init(const Settings& s) {
    settings_ = s;
    bitrateKbps_ = s.bitrateKbps;
    if (!CreateDevice(s) || !CreateRenderResources() || !CreateConverter()) {
        Log("PhoneVR: GPU pipeline initialisation failed");
        return false;
    }
    enc_ = std::make_unique<H264Encoder>();
    auto outFn = [this](const uint8_t* d, size_t n, bool key) {
        if (onPacket_) onPacket_(d, n, key, frameCounter_++, NowUs());
    };
    if (!enc_->Init(dev_.Get(), s.streamWidth, s.streamHeight, s.refreshRate, s.bitrateKbps, !s.useMainProfile, outFn))
        return false;
    stop_ = false;
    encThread_ = std::thread(&VideoPipeline::EncodeLoop, this);
    return true;
}

void VideoPipeline::Shutdown() {
    {
        std::lock_guard<std::mutex> lk(frameMutex_);
        stop_ = true;
    }
    frameCv_.notify_all();
    if (encThread_.joinable()) encThread_.join();
    if (enc_) { enc_->Shutdown(); enc_.reset(); }
    {
        std::lock_guard<std::mutex> lk(texMutex_);
        swapTex_.clear();
        syncTex_.clear();
    }
    for (auto& s : slots_) { s.view.Reset(); s.tex.Reset(); }
    vpIn_.Reset(); vp_.Reset(); vpEnum_.Reset();
    rgbRtv_.Reset(); rgbTex_.Reset(); blendOver_.Reset(); cbuf_.Reset(); sampler_.Reset(); ps_.Reset(); vs_.Reset();
    vctx_.Reset(); vdev_.Reset(); ctx_.Reset(); dev1_.Reset(); dev_.Reset();
}

void VideoPipeline::SetBitrate(uint32_t kbps) {
    kbps = std::clamp<uint32_t>(kbps, 1000, 200000);
    bitrateKbps_ = kbps;
    if (enc_) enc_->SetBitrate(kbps);
}

void VideoPipeline::RequestKeyframe() {
    if (enc_) enc_->ForceKeyframe();
}

// ---------------------------------------------------------------------------
// swap texture sets
// ---------------------------------------------------------------------------
void VideoPipeline::CreateSwapTextureSet(uint32_t pid, uint32_t w, uint32_t h, uint32_t fmt,
                                         uint32_t samples, uint64_t outHandles[3]) {
    (void)samples;  // the compositor hands us single-sample targets
    auto set = std::make_shared<SwapSet>();
    set->pid = pid;
    std::lock_guard<std::mutex> lk(texMutex_);
    for (int i = 0; i < 3; i++) {
        outHandles[i] = 0;
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = w;
        td.Height = h;
        td.MipLevels = td.ArraySize = 1;
        td.Format = (DXGI_FORMAT)fmt;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        td.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        SwapTex st;
        if (FAILED(dev_->CreateTexture2D(&td, nullptr, &st.tex))) {
            Log("PhoneVR: CreateTexture2D failed for swap texture (%ux%u fmt %u)", w, h, fmt);
            continue;
        }
        D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
        sv.Format = SrvFormatFor((DXGI_FORMAT)fmt, st.encodeSrgb);
        sv.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sv.Texture2D.MipLevels = 1;
        dev_->CreateShaderResourceView(st.tex.Get(), &sv, &st.srv);
        ComPtr<IDXGIResource> res;
        HANDLE sh = nullptr;
        if (FAILED(st.tex.As(&res)) || FAILED(res->GetSharedHandle(&sh)) || !sh) continue;
        st.set = set;
        st.index = i;
        set->handles[i] = (uint64_t)sh;
        outHandles[i] = (uint64_t)sh;
        swapTex_[(uint64_t)sh] = std::move(st);
    }
}

void VideoPipeline::DestroySwapTextureSet(uint64_t handle) {
    std::lock_guard<std::mutex> lk(texMutex_);
    auto it = swapTex_.find(handle);
    if (it == swapTex_.end()) return;
    auto set = it->second.set;
    for (uint64_t h : set->handles) swapTex_.erase(h);
}

void VideoPipeline::DestroyAllSwapTextureSets(uint32_t pid) {
    std::lock_guard<std::mutex> lk(texMutex_);
    for (auto it = swapTex_.begin(); it != swapTex_.end();) {
        if (it->second.set && it->second.set->pid == pid) it = swapTex_.erase(it);
        else ++it;
    }
}

void VideoPipeline::NextSwapTextureSetIndex(const uint64_t handles[2], uint32_t indices[2]) {
    std::lock_guard<std::mutex> lk(texMutex_);
    for (int i = 0; i < 2; i++) {
        auto it = swapTex_.find(handles[i]);
        indices[i] = it == swapTex_.end() ? 0 : (uint32_t)((it->second.index + 1) % 3);
    }
}

void VideoPipeline::SubmitLayer(const uint64_t handles[2], const float bounds[2][4]) {
    // Called once per layer, bottom to top, before each Present. Games with overlays
    // (DiRT Rally 2.0's HUD / fade panel, the SteamVR dashboard) submit several.
    std::lock_guard<std::mutex> lk(texMutex_);
    if (frameLayers_.size() >= 16) return;  // sanity cap
    Layer l;
    for (int i = 0; i < 2; i++) {
        l.handles[i] = handles[i];
        memcpy(l.bounds[i], bounds[i], sizeof(float) * 4);
    }
    frameLayers_.push_back(l);
}

// ---------------------------------------------------------------------------
// per-frame work
// ---------------------------------------------------------------------------
void VideoPipeline::DrawEye(int eye, const SwapTex* src, const float bounds[4], bool opaque) {
    if (!src || !src->srv) return;
    const float eyeW = settings_.streamWidth * 0.5f;
    const float H = (float)settings_.streamHeight;

    CBData cb = {};
    float shiftX = settings_.lensCenterShiftX;
    cb.lensCenter[0] = 0.5f + (eye == 0 ? shiftX : -shiftX);
    cb.lensCenter[1] = 0.5f + settings_.lensCenterShiftY;
    cb.scale[0] = 2.f * (eyeW / H);
    cb.scale[1] = 2.f;
    memcpy(cb.bounds, bounds, sizeof(cb.bounds));
    cb.k1 = settings_.lensK1;
    cb.k2 = settings_.lensK2;
    cb.chroma = settings_.chromatic;
    cb.encode = src->encodeSrgb ? 1.f : 0.f;
    cb.opaque = opaque ? 1.f : 0.f;

    D3D11_MAPPED_SUBRESOURCE m;
    if (FAILED(ctx_->Map(cbuf_.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &m))) return;
    memcpy(m.pData, &cb, sizeof(cb));
    ctx_->Unmap(cbuf_.Get(), 0);

    D3D11_VIEWPORT vp = {eye * eyeW, 0.f, eyeW, H, 0.f, 1.f};
    ctx_->RSSetViewports(1, &vp);
    ID3D11ShaderResourceView* srv = src->srv.Get();
    ctx_->PSSetShaderResources(0, 1, &srv);
    ctx_->Draw(3, 0);
    ID3D11ShaderResourceView* nullSrv = nullptr;
    ctx_->PSSetShaderResources(0, 1, &nullSrv);
}

void VideoPipeline::Present(uint64_t syncHandle) {
    if (!dev_) return;
    struct DrawLayer {
        SwapTex eyes[2];
        float bounds[2][4];
    };
    std::vector<DrawLayer> layers;
    ComPtr<ID3D11Texture2D> sync;
    size_t submitted = 0;
    {
        std::lock_guard<std::mutex> lk(texMutex_);
        if (!frameLayers_.empty()) {
            lastLayers_.swap(frameLayers_);
            frameLayers_.clear();
        }
        submitted = lastLayers_.size();
        for (const Layer& l : lastLayers_) {
            DrawLayer d;
            bool ok = true;
            for (int i = 0; i < 2 && ok; i++) {
                auto it = swapTex_.find(l.handles[i]);
                if (it == swapTex_.end()) { ok = false; break; }
                d.eyes[i] = it->second;
                memcpy(d.bounds[i], l.bounds[i], sizeof(d.bounds[i]));
            }
            if (ok) layers.push_back(std::move(d));  // skip layers whose textures are gone
        }
        if (syncHandle) {
            auto it = syncTex_.find(syncHandle);
            if (it != syncTex_.end()) {
                sync = it->second;
            } else {
                ComPtr<ID3D11Texture2D> t;
                if (SUCCEEDED(dev_->OpenSharedResource((HANDLE)syncHandle, IID_PPV_ARGS(&t))) ||
                    (dev1_ && SUCCEEDED(dev1_->OpenSharedResource1((HANDLE)syncHandle, IID_PPV_ARGS(&t))))) {
                    syncTex_[syncHandle] = t;
                    sync = t;
                }
            }
        }
    }
    if (submitted != loggedLayerCount_) {
        Log("PhoneVR: compositor now submits %zu layer(s) per frame", submitted);
        loggedLayerCount_ = submitted;
    }
    if (layers.empty()) return;

    ComPtr<IDXGIKeyedMutex> km;
    if (sync) sync.As(&km);
    if (km && km->AcquireSync(0, 10) != S_OK) return;  // compositor still owns it; skip frame

    // --- lens distortion into the side-by-side RGB frame ---
    const float black[4] = {0, 0, 0, 1};
    ID3D11RenderTargetView* rtv = rgbRtv_.Get();
    ctx_->OMSetRenderTargets(1, &rtv, nullptr);
    ctx_->ClearRenderTargetView(rtv, black);
    ctx_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx_->IASetInputLayout(nullptr);
    ctx_->VSSetShader(vs_.Get(), nullptr, 0);
    ctx_->PSSetShader(ps_.Get(), nullptr, 0);
    ID3D11Buffer* cb = cbuf_.Get();
    ctx_->PSSetConstantBuffers(0, 1, &cb);
    ID3D11SamplerState* sp = sampler_.Get();
    ctx_->PSSetSamplers(0, 1, &sp);
    // Bottom layer (the game's scene) opaque, every further layer alpha-blended on top.
    for (size_t li = 0; li < layers.size(); li++) {
        const bool opaque = li == 0;
        ctx_->OMSetBlendState(opaque ? nullptr : blendOver_.Get(), nullptr, 0xffffffff);
        for (int e = 0; e < 2; e++) DrawEye(e, &layers[li].eyes[e], layers[li].bounds[e], opaque);
    }
    ctx_->OMSetBlendState(nullptr, nullptr, 0xffffffff);
    ctx_->OMSetRenderTargets(0, nullptr, nullptr);

    // --- pick a free NV12 slot and convert ---
    int slot = -1;
    {
        std::lock_guard<std::mutex> lk(frameMutex_);
        for (int i = 0; i < kSlots; i++) {
            if (!slots_[i].busy) { slot = i; break; }
        }
    }
    if (slot >= 0) {
        D3D11_VIDEO_PROCESSOR_STREAM st = {};
        st.Enable = TRUE;
        st.pInputSurface = vpIn_.Get();
        if (SUCCEEDED(vctx_->VideoProcessorBlt(vp_.Get(), slots_[slot].view.Get(), 0, 1, &st))) {
            ctx_->Flush();
            if (km) km->ReleaseSync(0);
            km.Reset();
            {
                std::lock_guard<std::mutex> lk(frameMutex_);
                if (pendingSlot_ >= 0) slots_[pendingSlot_].busy = false;  // encoder is behind: drop oldest
                pendingSlot_ = slot;
                slots_[slot].busy = true;
            }
            frameCv_.notify_one();
        }
    }
    ctx_->Flush();
    if (km) km->ReleaseSync(0);
}

void VideoPipeline::EncodeLoop() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    const double usPerFrame = 1e6 / settings_.refreshRate;
    uint64_t n = 0;
    for (;;) {
        int idx;
        {
            std::unique_lock<std::mutex> lk(frameMutex_);
            frameCv_.wait(lk, [&] { return stop_ || pendingSlot_ >= 0; });
            if (stop_) return;
            idx = pendingSlot_;
            pendingSlot_ = -1;
        }
        enc_->Encode(slots_[idx].tex.Get(), (uint64_t)(n++ * usPerFrame));
        std::lock_guard<std::mutex> lk(frameMutex_);
        slots_[idx].busy = false;
    }
}

}  // namespace phonevr
