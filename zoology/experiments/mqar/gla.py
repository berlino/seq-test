from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig


vocab_size = 8_192
seq_len = 512
num_kv_pairs = 64
d_model = 64

config = TrainConfig(
    data=DataConfig(
        # cache_dir="/path/to/cache/dir"  TODO: add this
        vocab_size=8_192,
        input_seq_len=seq_len,
        num_train_examples=100_000,
        num_test_examples=3_000,
        builder=FunctionConfig(
            name="zoology.data.associative_recall.multiquery_ar",
            kwargs={
                "num_kv_pairs": num_kv_pairs,
                "train_power_a": 0.01,
                "test_power_a": 0.01,
                "random_non_queries": False
            }
        ),
        
    ),
    model=ModelConfig(
        d_model=d_model,
        n_layers=4,
        block_type="TransformerBlock",
        vocab_size=vocab_size,
        max_position_embeddings=seq_len,
        sequence_mixer=ModuleConfig(
            name="fla.layers.gla.GatedLinearAttention",
            kwargs={
                        "mode": "fused_recurrent",
                        "num_heads": 2,
                        'use_gk': True,
                        "use_gv": False,
                        "gate_logit_normalizer": 16,
                    }              
        ),
        state_mixer=dict(name="torch.nn.Identity", kwargs={}),
    ),

    learning_rate=1e-3,
    max_epochs=64,
    run_id=f"mqar_seq{seq_len}_kv{num_kv_pairs}_mha_dmodel{d_model}",
    logger=LoggerConfig(
        project_name="seq-test",
    )
)


configs = [config]