from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig


vocab_size = 18
model_vocab_size = 18 + 1 # the additional seperator token
seq_len = 512
d_model = 128

config = TrainConfig(
    data=DataConfig(
        vocab_size=vocab_size,
        input_seq_len=seq_len,
        num_train_examples=5_000,
        num_test_examples=1_000,
        builder=FunctionConfig(
            name="zoology.data.regbench.regbench",
            kwargs={}
        ),
        
    ),
    model=ModelConfig(
        d_model=d_model,
        n_layers=4,
        block_type="TransformerBlock",
        vocab_size=model_vocab_size,
        max_position_embeddings=seq_len,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA",
            kwargs={"dropout": 0.1, "num_heads": 2}
        ),
        state_mixer=ModuleConfig(
            name="zoology.mixers.mlp.SwiGLU", 
            kwargs={"intermediate_size": 320}
        )
    ),

    learning_rate=2.54e-4,
    max_epochs=200,
    run_id=f"regbench_seq{seq_len}_vocab{vocab_size}_mha_dmodel{d_model}",
    logger=LoggerConfig(
        project_name="seq-test",
    )
)


configs = [config]