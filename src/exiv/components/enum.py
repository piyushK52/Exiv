from websockets import ExtensionName
from ..utils.enum import ExtendedEnum


class KSamplerType(ExtendedEnum):
    EULER = "euler"
    EULER_CFG_PP = "euler_cfg_pp"
    EULER_ANCESTRAL = "euler_ancestral"
    EULER_ANCESTRAL_CFG_PP = "euler_ancestral_cfg_pp"
    HEUN = "heun"
    HEUNPP2 = "heunpp2"
    DPM_2 = "dpm_2"
    DPM_2_ANCESTRAL = "dpm_2_ancestral"
    LMS = "lms"
    DPM_FAST = "dpm_fast"
    DPM_ADAPTIVE = "dpm_adaptive"
    DPMPP_2S_ANCESTRAL = "dpmpp_2s_ancestral"
    DPMPP_SDE = "dpmpp_sde"
    DPMPP_SDE_GPU = "dpmpp_sde_gpu"
    DPMPP_2M = "dpmpp_2m"
    DPMPP_2M_SDE = "dpmpp_2m_sde"
    DPMPP_2M_SDE_GPU = "dpmpp_2m_sde_gpu"
    DPMPP_3M_SDE = "dpmpp_3m_sde"
    DPMPP_3M_SDE_GPU = "dpmpp_3m_sde_gpu"
    DDPM = "ddpm"
    LCM = "lcm"
    IPNDM = "ipndm"
    IPNDM_V = "ipndm_v"
    DEIS = "deis"

class SamplerType(ExtendedEnum):
    DDIM = "ddim"
    UNI_PC = "uni_pc"
    UNI_PC_BH2 = "uni_pc_bh2"
    
'''
this is like a trick of the trade, a1111 issue - https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/15050
basically the last jump from very low noise to 0 noise in ksampler is numerically unstable
for certain samplers, and thus we kind of widen the jump, from a slightly bigger number to 0
'''
DISCARD_PENULTIMATE_SIGMA_SAMPLERS = [
    KSamplerType.DPM_2.value, 
    KSamplerType.DPM_2_ANCESTRAL.value, 
    SamplerType.UNI_PC.value, 
    SamplerType.UNI_PC_BH2.value,
]
    
class SchedulerType(ExtendedEnum):
    NORMAL = "normal"
    KARRAS = "karras"
    EXPONENTIAL = "exponential"
    SGM_UNIFORM = "sgm_uniform"
    SIMPLE = "simple"
    DDIM_UNIFORM = "ddim_uniform"
    BETA = "beta"
    

class BetaSchedule(ExtendedEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    SQRT_LINEAR = "sqrt_linear"
    SQRT = "sqrt"
    
class ModelType(ExtendedEnum):
    EPS = "eps"
    V_PREDICTION = "v_pred"
    V_PREDICTION_EDM = "v_pred_edm"
    FLOW = "flow"
    STABLE_CASCADE = "stable_cascade"
    EDM = "edm"
    V_PREDICTION_CONTINUOUS = "v_pred_cont"
    FLUX = "flux"

class TextEncoderType(ExtendedEnum):
    CLIP_G = "clip_g"
    CLIP_H = "clip_h"
    CLIP_L = "clip_l"

    T5_XXL = "t5_xxl"
    UMT5_XXL = "umt5_xxl"
    T5_XL = "t5_xl"
    T5_XXL_OLD = "t5_xxl_old"
    T5_BASE = "t5_base"
    BYT5_SMALL_GLYPH = "byt5_small_glyph"

    GEMMA_2_2B = "gemma_2_2b"
    QWEN25_3B = "qwen25_3b"
    QWEN25_7B = "qwen25_7b"
    LLAMA3_8 = "llama3_8"
    QWEN3_EMBEDDING_0_6B = "qwen3_embedding_0_6b"

class VisionEncoderType(ExtendedEnum):
    CLIP_G = "clip_g"
    CLIP_H = "clip_h"
    CLIP_L = "clip_l"
    CLIP_L_336 = "clip_l_336"
    CLIP_L_LLAVA = "clip_l_llava"
    
    SIGLIP_384 = "siglip_384"
    SIGLIP_512 = "signlip_512"
    
    DINO2_L = "dino2_l"
    DINO2_G = "dino2_g"
    

class Model(ExtendedEnum):
    # ------------- WAN Models -------------
    WAN22_5B_T2V = "wan22_5b_t2v"               # supports i2v through inpainting
    WAN22_14B_TI2V = "wan22_14b_ti2v"           # takes in a separate ref latent
    
    WAN21_1_3B_T2V = "wan21_1_3b_t2v"           # no native i2v support
    WAN21_14B_TI2V = "wan21_14b_ti2v"           # basically wan21
    
    # supports multiple ref types
    WAN21_VACE_1_3B_R2V = "wan21_1_3b_vace_r2v"
    WAN21_VACE_14B_R2V = "wan21_14b_vace_r2v"
    
    WAN22_14B_ANIMATE = "wan22_14b_animate"

    # ------------- QWEN Models -------------
    QWEN_IMAGE = "qwen_image"
    QWEN3_TTS_VOICE_DESIGN = "qwen3_tts_voice_design"
    QWEN3_TTS_CUSTOM_VOICE = "qwen3_tts_custom_voice"
    QWEN3_TTS_BASE = "qwen3_tts_base"

    # ------------- ACE-Step Models -------------
    ACESTEP_V15_SFT = "acestep_v15_sft"
    ACESTEP_V15_TURBO = "acestep_v15_turbo"
    ACESTEP_5HZ_LM_1_7B = "acestep_5hz_lm_1_7b"

class VAEType(ExtendedEnum):
    WAN21 = "wan21"
    WAN22 = "wan22"
    ACESTEP_OOBLECK = "acestep_oobleck"