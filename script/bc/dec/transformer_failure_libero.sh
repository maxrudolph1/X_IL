python run.py --config-name=libero_failure_config \
            --multirun agents=bc_agent \
            agent_name=bc_transformer_failure_inclusive \
            group=bc_decoder_only_failure_inclusive \
            agents/model=bc/bc_dec_transformer \
            seed=0,1,2
