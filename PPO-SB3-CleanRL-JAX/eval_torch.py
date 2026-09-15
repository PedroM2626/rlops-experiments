def greedy_torch_logits(agent, obs):
    import numpy as _np
    import torch
    with torch.no_grad():
        logits = agent.actor(
            torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    return _np.asarray(logits[0])


def eval_torch_pt(env_id, model_path, n_ep):
    import sys
    import numpy as _np
    import config as _c
    sys.path.insert(0, str(_c.BASE_DIR))
    import torch
    import torch.nn as nn
    import gymnasium as gym
    from train_cleanrl_torch import build_agent
    from specs import SPEC_CLEANRL
    env0 = gym.make(env_id)
    od = int(_np.prod(env0.observation_space.shape))
    ad = int(env0.action_space.n)
    env0.close()
    agent = build_agent(nn, od, ad, SPEC_CLEANRL)
    agent.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.eval()
    out = []
    for i in range(n_ep):
        env = gym.make(env_id)
        obs, _ = env.reset(seed=_c.EVAL_SEED_OFFSET + i)
        tot, done = 0.0, False
        while not done:
            logits = greedy_torch_logits(agent, _np.asarray(obs,
                                                             dtype=_np.float32))
            obs, r, term, trunc, _ = env.step(int(_np.argmax(logits)))
            tot += float(r)
            done = bool(term or trunc)
        out.append(tot)
        env.close()
    return out
