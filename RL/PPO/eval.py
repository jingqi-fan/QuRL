




class parallel_eval():
    def __init__(self, model, eval_env, eval_freq, eval_t, test_policy, test_seed, init_test_queues, test_batch, device, num_pool, time_f, policy_name, per_iter_normal_obs, env_config_name, bc, randomize = True,
                 verbose=1):
        super(parallel_eval, self).__init__(verbose)
        self.model = model