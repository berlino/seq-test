import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig

vocab_size = 8_192
seq_len = 512
num_kv_pairs = 64
d_model = 128

factory_kwargs = {
    "num_kv_pairs": num_kv_pairs,
    "train_power_a": 0.01,
    "test_power_a": 0.01,
    "random_non_queries": False
}

configs = []
for lr in  np.logspace(-4, -2, 4):
    config = TrainConfig(
        data=DataConfig(
            train_configs=[MQARConfig(num_examples=100_000, vocab_size=vocab_size, input_seq_len=seq_len, **factory_kwargs)],
            test_configs=[MQARConfig(num_examples=3_000, vocab_size=vocab_size, input_seq_len=seq_len, **factory_kwargs)],
            batch_size=128,
            # cache_dir="/var/cr05_data/sabri_data/zoology",
        ),
        model=ModelConfig(
            d_model=d_model,
            n_layers=4,
            block_type="TransformerBlock",
            vocab_size=vocab_size,
            max_position_embeddings=-1,
            sequence_mixer=ModuleConfig(
                name="fla.layers.delta_net.DeltaNet",
                kwargs={
                        "hidden_size:": d_model,
                        "num_heads": 2,
                        "use_short_conv": False,
                    }              
            ),
            state_mixer=dict(name="torch.nn.Identity", kwargs={}),
        ),

        learning_rate=lr,
        max_epochs=64,
        run_id=f"mqar_seq{seq_len}_kv{num_kv_pairs}_delta_dmodel{d_model}",
        logger=LoggerConfig(
            project_name="seq-test",
            entity="bailin"
        ),
        early_stopping_patience=32,
    )
    configs.append(config)