import yaml
import random

# ---------- 配置（硬编码） ----------
N = 200
train_T = 5000
test_T = 5000
p_conn = 0.5  # network 联通(1)的概率；不联通(0)概率=0.3

random.seed(42)  # 如不需要确定性，可删除

# ---------- 仅让指定集合用“流式”输出 ----------
class FlowList(list): ...
class FlowDict(dict): ...
class FlowDumper(yaml.SafeDumper): ...

def _repr_flow_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

def _repr_flow_dict(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=True)

FlowDumper.add_representer(FlowList, _repr_flow_list)
FlowDumper.add_representer(FlowDict, _repr_flow_dict)

# ---------- 生成 network ----------
# 按 p_conn 生成 N×N 0/1 矩阵
network = FlowList([
    FlowList([1 if random.random() < p_conn else 0 for _ in range(N)])
    for _ in range(N)
])

# ---------- 生成 mu（与 network 对应） ----------
# 若 network[i][j] == 0 -> mu[i][j] = 0
# 若 network[i][j] == 1 -> mu[i][j] ~ U(0,1)
mu = FlowList([
    FlowList([random.random() if network[i][j] == 1 else 0 for j in range(N)])
    for i in range(N)
])

# ---------- 其他字段 ----------
lam_val = FlowList([random.uniform(0.001, 2.0) for _ in range(N)])
h = FlowList([random.random() for _ in range(N)])
init_queues = FlowList([0 for _ in range(N)])

data = {
    "name": "n_model_large",
    "lam_type": "constant",
    "lam_params": FlowDict({"val": lam_val}),
    "network": network,
    "mu": mu,
    "h": h,
    "init_queues": init_queues,
    "queue_event_options": None,
    "train_T": train_T,
    "test_T": test_T,
    "num_pool": 1,
}

# ---------- 输出（顶层块状，内部矩阵单行） ----------
output_path = f"../env/en_model_mm_{N}.yaml"
with open(output_path, "w", encoding="utf-8") as f:
    yaml.dump(
        data,
        f,
        Dumper=FlowDumper,
        sort_keys=False,
        default_flow_style=False,  # 顶层块状
        width=1_000_000,           # 避免换行
        allow_unicode=True,
    )

print(f"已生成 {output_path}")

