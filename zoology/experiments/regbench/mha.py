import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig
from zoology.data.regbench import RegBenchConfig

vocab_size = 18
model_vocab_size = 18 + 2 # the additional seperator token
seq_len = 512
d_model = 256 # 128
swiglu_intermediate_size = 512 # 320

configs = []
for num_epoch in [30, 60, 90, 120]:
    config = TrainConfig(
        data=DataConfig(
            train_configs=[RegBenchConfig(vocab_size=vocab_size, input_seq_len=seq_len, num_examples=10_000)],
            test_configs=[RegBenchConfig(vocab_size=vocab_size, input_seq_len=seq_len, num_examples=1_000)],
            batch_size=32,
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
                kwargs={"intermediate_size": swiglu_intermediate_size}  
            )
        ),

        learning_rate=3e-4,
        max_epochs=num_epoch,
        run_id=f"regbench_seq{seq_len}_vocab{vocab_size}_mha_dmodel{d_model}_epoch{num_epoch}",
        logger=LoggerConfig(
            project_name="seq-test",
            entity="bailin"
        )
    )

    configs.append(config)
