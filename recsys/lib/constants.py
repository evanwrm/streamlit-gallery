# Default Columns
DEFAULT_COLS_USER = ["userID", "author.steamid"]
DEFAULT_COLS_ITEM = ["itemID", "app_id", "appid"]
DEFAULT_COLS_RATING = ["rating", "recommended"]
DEFAULT_COLS_TIMESTAMP = ["timestamp", "timestamp_created", "timestamp_updated"]

DEFAULT_COL_USER = DEFAULT_COLS_USER[0]
DEFAULT_COL_ITEM = DEFAULT_COLS_ITEM[0]
DEFAULT_COL_RATING = DEFAULT_COLS_RATING[0]
DEFAULT_COL_TIMESTAMP = DEFAULT_COLS_TIMESTAMP[0]
DEFAULT_COL_PREDICTIONS = "prediction"

# Features
DEFAULT_BIN_COUNT = 5
MAX_BIN_COUNT = 100

DEFAULT_DELIMITER = ";"

# Default Retrieval
DEFAULT_TOP_K = 10

# Model IDs
BPR_ID = "bpr"
NRMS_ID = "nrms"
BIVAE_ID = "bivae"
SLI_REC_ID = "sli_rec"
SUM_ID = "sum"
GRU4REC_ID = "gru4rec"
LIGHTFM_ID = "lightfm"
LIGHTGCN_ID = "lightgcn"
VOWPAL_WABBIT_ID = "vowpal_wabbit"
WIDE_AND_DEEP_ID = "wide_and_deep"

# Activation functions
ACT_SIGMOID = "sigmoid"
ACT_TANH = "tanh"
ACT_ELU = "elu"
ACT_RELU = "relu"
ACT_RELU6 = "relu6"

# Loss functions
LOSS_WARP = "warp"
LOSS_LOGISTIC = "logistic"
LOSS_BPR = "bpr"
LOSS_WARP_KOS = "warp-kos"

# Optimizers
OPT_ADADELTA = "adadelta"
OPT_ADAGRAD = "adagrad"

# Likelihoods
LIKELIHOOD_BERN = "bern"
LIKELIHOOD_GAUS = "gaus"
LIKELIHOOD_POIS = "pois"
