import yaml
import random

# ---------- 配置（硬编码） ----------
N = 10
train_T = 50000
test_T = 50000
p_conn = 0.7  # network 联通(1)的概率；不联通(0)概率=0.3

# 生成 G/G/N 的“Gaussian相关”参数（截断正态的均值/方差）
# 这里我们随机生成每个队列的到达/服务的均值；你也可以改成固定常数或自定义数组
ARR_MEAN_MIN, ARR_MEAN_MAX = 0.2, 1   # 到达间隔均值（越大 -> 平均间隔更长 -> 到达更稀疏）
ARR_STD = 0.35                           # 到达间隔标准差（截断正态的 std）
SRV_MEAN_MIN, SRV_MEAN_MAX = 0.2, 1   # 服务“工作量”均值（越大 -> 平均工作量更大）
SRV_STD = 0.5                            # 服务工作量标准差

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
# mu = FlowList([
#     FlowList([random.random() if network[i][j] == 1 else 0 for j in range(N)])
#     for i in range(N)
# ])
mu = FlowList([
    FlowList([1 if network[i][j] == 1 else 0 for j in range(N)])
    for i in range(N)
])



# ---------- 其他字段 ----------
# G/G/N 到达与服务的“Gaussian相关”配置（截断正态）
arrival_mean = FlowList([random.uniform(ARR_MEAN_MIN, ARR_MEAN_MAX) for _ in range(N)])
service_mean = FlowList([random.uniform(SRV_MEAN_MIN, SRV_MEAN_MAX) for _ in range(N)])

arrival_dist = FlowDict({
    "type": "truncnorm",         # 截断正态，>0
    "mean": arrival_mean,        # 每个队列的到达间隔均值
    "std":  ARR_STD              # 可用标量或长度为 N 的列表
})

service_dist = FlowDict({
    "type": "truncnorm",         # 截断正态，>0
    "mean": service_mean,        # 每个队列的服务“工作量”均值（不含 mu）
    "std":  SRV_STD
})

# 成本向量等
h = FlowList([random.random() for _ in range(N)])
init_queues = FlowList([0 for _ in range(N)])

data = {
    "name": "n_model_ggn",
    "arrival_dist": arrival_dist,   # <- 新增：G 到达（Gaussian相关）
    "service_dist": service_dist,   # <- 新增：G 服务（Gaussian相关，工作量）
    # 下面与原格式一致
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
output_path = f"../env/n_model_gg_{N}.yaml"
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
