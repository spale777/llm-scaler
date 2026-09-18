"""DeepSeek-V4.1-Flash architecture constants, from the published config.json.

Every kernel targeting this model is dimensioned by these numbers, and the
failure mode when one is wrong is silent: a router that groups experts the
config never groups still returns num_experts_per_tok of them, and a head_dim
sourced from a different DeepSeek generation still produces a full output tile.

Values are transcribed from deepseek-ai/DeepSeek-V4.1-Flash config.json
(text_config unless noted). Fields that are ABSENT are recorded as such,
because their absence is itself load-bearing: no n_group / topk_group means
noaux_tc selection ranges over all experts.
"""

# --- attention -------------------------------------------------------------
NUM_HIDDEN_LAYERS = 40
HIDDEN_SIZE = 5120
NUM_ATTENTION_HEADS = 64
NUM_KEY_VALUE_HEADS = 1          # MQA: one KV head shared by all 64 query heads
HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
Q_LORA_RANK = 1280
O_LORA_RANK = 1024
O_GROUPS = 8                     # block-diagonal wo_a
SLIDING_WINDOW = 128
RMS_NORM_EPS = 1e-20

# --- CSA2: compressed sparse attention -------------------------------------
# One entry per layer plus the trailing three; 0 = full, 1 and 2 are the
# compression ratios whose layers share a compressed KV cache and one indexer.
COMPRESS_RATIOS = [0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0]
COMPRESS_ROPE_THETA = 160000
# Only these layers produce main KV; the rest reuse it.
KV_SOURCE_LAYER_IDS = [2, 8, 14, 20]
INDEX_SOURCE_LAYER_IDS = [2, 8, 14, 20, 24, 28, 32, 36]

# --- lightning indexer ------------------------------------------------------
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_TOPK = 512
# Hierarchical stage: one layer builds the candidate pool every later indexing
# layer is restricted to.
CANDIDATE_SOURCE_LAYER_ID = 20
CANDIDATE_TOPK_BLOCKS = 2048
CANDIDATE_BLOCK_SIZE = 8

# --- MoE --------------------------------------------------------------------
N_ROUTED_EXPERTS = 384
N_SHARED_EXPERTS = 1
NUM_EXPERTS_PER_TOK = 6
MOE_INTERMEDIATE_SIZE = 2304
SCORING_FUNC = "sqrtsoftplus"
TOPK_METHOD = "noaux_tc"
NORM_TOPK_PROB = True
ROUTED_SCALING_FACTOR = 1.5
SWIGLU_LIMIT = 10.0
# Absent from config.json. noaux_tc here is ungrouped; a group-limited stage
# would mask experts this model never masks.
N_GROUP = None
TOPK_GROUP = None

# --- quantization -----------------------------------------------------------
QUANT_METHOD = "fp8"
ACTIVATION_SCHEME = "dynamic"
WEIGHT_BLOCK_SIZE = [32, 32]     # not 128x128
SCALE_FMT = "ue8m0"
EXPERT_DTYPE = "fp4"             # E2M1

# --- hyper-connections (already native on XPU) ------------------------------
HC_MULT = 4
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-06

# --- engram -----------------------------------------------------------------
ENGRAM_LAYER_IDS = [1, 14]
ENGRAM_MAX_NGRAM_SIZE = 4
ENGRAM_VOCAB_SIZE = 16000000
ENGRAM_N_HEADS = 8
ENGRAM_HEAD_DIM = 256
ENGRAM_COMPRESSED_VOCAB_SIZE = 99092
ENGRAM_PAD_TOKEN_ID = 2

# --- DSpark MTP (deferred past v1) ------------------------------------------
NUM_NEXTN_PREDICT_LAYERS = 3
DSPARK_BLOCK_SIZE = 5
DSPARK_TARGET_LAYER_IDS = [37, 38, 39]
DSPARK_N_ROUTED_EXPERTS = 128
DSPARK_NUM_EXPERTS_PER_TOK = 3

# --- rope -------------------------------------------------------------------
ROPE_THETA = 10000
ROPE_SCALING = {
    "rope_type": "yarn",
    "factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "original_max_position_embeddings": 65536,
}
MAX_POSITION_EMBEDDINGS = 1048576
