import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, FunctionConfig, ModuleConfig, LoggerConfig
from zoology.data.regbench import RegBenchConfig

vocab_size = 18
model_vocab_size = 18 + 2 # the additional seperator token
seq_len = 512
d_model = 128


configs = []
# for num_epoch in [30, 60, 90, 120]:
for num_epoch in [90]:
    config = TrainConfig(
        data=DataConfig(
            train_configs=[RegBenchConfig(vocab_size=vocab_size, input_seq_len=seq_len, num_examples=5_000)],
            test_configs=[RegBenchConfig(vocab_size=vocab_size, input_seq_len=seq_len, num_examples=1_000)],
            batch_size=32,
        ),
        model=ModelConfig(
            d_model=d_model,
            n_layers=4,
            block_type="TransformerBlock",
            vocab_size=model_vocab_size,
            max_position_embeddings=-1,
            sequence_mixer=ModuleConfig(
                name="fla.layers.gla.GatedLinearAttention",
                kwargs={
                        "hidden_size:": d_model,
                        "mode": "fused_recurrent",
                        "num_heads": 2,
                        'use_gk': True,
                        "use_gv": False,
                        "gate_logit_normalizer": 16,
                    }              
            ),
            state_mixer=ModuleConfig(
                name="zoology.mixers.mlp.SwiGLU", 
                kwargs={"intermediate_size": 320}
            )
        ),

        learning_rate=2.54e-4, # from icll paper
        max_epochs=num_epoch,
        run_id=f"regbench_seq{seq_len}_vocab{vocab_size}_gla_dmodel{d_model}_epoch{num_epoch}",
        logger=LoggerConfig(
            project_name="seq-test",
            entity="bailin"
        ),
        early_stopping_patience=32,
    )
    configs.append(config)