import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig

vocab_size = 8_192

# seq_len = 1024
# num_kv_pairs = 64
# d_model = 128

seq_len = 512
num_kv_pairs = 64
d_model = 128
# d_model = 64

# seq_len = 256
# num_kv_pairs = 16
# d_model = 128

# seq_len = 128
# num_kv_pairs = 8
# d_model = 64


num_head = 1
expand_k = 1
expand_v = 1
dk_per_head = d_model * expand_k // num_head
dv_per_head = d_model * expand_v // num_head
dv_per_head_to_logit_normalizer = {512: 64, 256: 32, 128: 32, 64: 16, 32: 16, 16: 16}
gate_logit_normalizer = dv_per_head_to_logit_normalizer[dv_per_head]

factory_kwargs = {
    "num_kv_pairs": num_kv_pairs,
    "train_power_a": 0.01,
    "test_power_a": 0.01,
    "random_non_queries": False
}

configs = []
for lr in np.logspace(-4, -2, 4):
# for lr in [0.002154434690031882, 0.0005]:
# for lr in [0.01]:
    config = TrainConfig(
        data=DataConfig(
            train_configs=[MQARConfig(num_examples=100_000, vocab_size=vocab_size, input_seq_len=seq_len, **factory_kwargs)],
            test_configs=[MQARConfig(num_examples=3_000, vocab_size=vocab_size, input_seq_len=seq_len, **factory_kwargs)],
            # batch_size=256,
            batch_size=128,
            # batch_size=64,
            # cache_dir="/var/cr05_data/sabri_data/zoology",
        ),
        model=ModelConfig(
            d_model=d_model,
            n_layers=4,
            block_type="TransformerBlock",
            vocab_size=vocab_size,
            max_position_embeddings=-1,
            sequence_mixer=ModuleConfig(
                name="zoology.mixers.rms.RMSLinearAttention",
                kwargs={"num_heads": num_head, "expand_k": expand_k, "expand_v": expand_v, "gate_logit_normalizer": gate_logit_normalizer}
            ),
            state_mixer=dict(name="torch.nn.Identity", kwargs={}),
        ),

        learning_rate=lr,
        run_id=f"mqar_seq{seq_len}_kv{num_kv_pairs}_rms_dmodel{d_model}",
        logger=LoggerConfig(
            project_name="seq-test",
            entity="bailin",
        ),
        # max_epochs=128,
        # early_stopping_patience=128,
        max_epochs=64,
        early_stopping_patience=64,
    )
    configs.append(config)