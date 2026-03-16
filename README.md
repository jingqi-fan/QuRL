# QuRL

**A Fast GPU-Accelerated Reinforcement Learning Framework for Large-Scale Queuing Networks**

QuRL is a GPU-accelerated reinforcement learning framework for simulating and controlling large-scale queuing networks. It provides efficient parallel simulation and a modular RL interface to support both classical queueing policies and modern reinforcement learning algorithms.

The framework is designed for scalable experimentation in complex queuing systems, including cloud computing, manufacturing systems, and traffic networks.

Paper:


---

# Overview

Queuing networks are widely used to model real-world systems such as cloud computing clusters, supply chains, and transportation systems. Reinforcement learning methods have recently been applied to these systems but often suffer from poor scalability due to inefficient environment simulation. 

QuRL addresses this challenge by providing a **fully GPU-accelerated simulation environment** that supports large-scale parallel training.

Key ideas include:

* Fully tensorized environment implementation
* Internally batched simulation on GPUs
* Efficient integration with RL algorithms
* Flexible configuration of queuing network structures

Experiments show that QuRL achieves **significant speedups over CPU-based frameworks** for large-scale systems. 

---

# Key Features

### GPU-Accelerated Simulation

QuRL stores environment states directly on the GPU using TensorDicts, eliminating repeated CPU–GPU communication overhead.

### High Parallelism

The framework supports **internally batched environments**, allowing many simulation trajectories to be executed simultaneously on the GPU.

### Modular Design

Users can easily extend the framework with:

* new queueing environments
* new control policies
* custom reinforcement learning algorithms

### Flexible Network Modeling

QuRL supports multiple types of queuing network structures including:

* N model
* X model
* extended N model
* reentrant queuing networks

These models represent different real-world service system structures. 

---

# Framework Architecture

The framework consists of three major components:

### Environment

Simulates discrete-event queuing dynamics including:

* job arrivals
* service completion
* routing decisions

Each simulation step advances to the next event determined by the minimum residual time.

### Policy

Policies determine how servers are assigned to queues.

Supported policies include:

**Classical policies**

* c-µ policy
* MaxWeight
* MaxPressure

**Reinforcement learning algorithms**

* PPO
* A2C
* PPO with behavior cloning initialization
* Pathwise policy gradient

### Internal Parallel Simulation

Instead of launching multiple environments, QuRL runs **multiple trajectories inside a single environment instance** using batched tensor operations.

This approach greatly improves performance and avoids Python overhead. 

---

# Supported Queuing Models

QuRL includes several predefined queuing network environments.

### N Model

Two queues and two servers where one server may serve both queues.

### X Model

A more flexible structure where both servers can serve both queues.

### Extended N Model

Generalized system with arbitrary numbers of queues and servers.

### Reentrant Model

Jobs may re-enter the system after service completion with a given probability.

These models allow the study of complex scheduling and routing policies in large-scale systems. 

---

# Installation

```bash
git clone https://github.com/jingqi-fan/QuRL.git
cd QuRL
pip install -r requirements.txt
```

Recommended environment:

```
Python >= 3.9
PyTorch >= 2.0
TorchRL
CUDA >= 11
```

---

# Quick Start

Example: run a training experiment

```bash
python train.py --config configs/env_data/reentrant_2.yaml \
                --policy RL/policy_configs/a2c.yaml
```

Configuration files control:

* environment settings
* network topology
* arrival distributions
* RL algorithm hyperparameters

---

# Configuration

Environment settings are defined in YAML files.

Example:

```
queue_num: 10
server_num: 10
max_job_num: 100
arrival_distribution: poisson
service_distribution: exponential
```

Policy configuration files define:

```
learning_rate
batch_size
discount_factor
training_iterations
```

This design allows experiments to be reproduced and modified easily.

---

# Experiments

Experiments compare QuRL with existing CPU-based frameworks such as QGym.

Results show:

* QuRL maintains stable runtime even when system size grows.
* CPU-based simulation time increases rapidly with the number of queues and servers.

In large systems, QuRL achieves **over 90% runtime reduction** compared with prior frameworks. 

---

# Evaluation Metrics

Performance is evaluated using:

* average queue length
* end-to-end runtime
* policy performance under different system scales

Average queue length is defined as

```
E[Q] = E[ (1/n) Σ_i ∫ Q_i(t) dt ]
```

This metric reflects overall system congestion. 

---

# Citation

If you use QuRL in your research, please cite:

```
@article{qurl2022,
  title={QuRL: A Fast GPU-Accelerated Reinforcement Learning Framework for Large-Scale Queuing Networks},
  journal={Journal of Machine Learning Research},
  year={2022}
}
```

---

# License

This project is released under the **CC BY 4.0 license**. 

---

# Contributing

Contributions are welcome.

Possible directions include:

* new queuing environments
* additional RL algorithms
* benchmarking tools
* visualization utilities

Please open an issue or submit a pull request.

---

如果你愿意，我还可以帮你做 **三个能显著提升 GitHub 项目质量的改进**：

1️⃣ 写一个 **更吸引人的 README 开头（很多 ML 项目会用的“亮点版本”）
2️⃣ 加一个 **Framework Architecture 图（README 常见）**
3️⃣ 帮你写 **Project Structure 部分**（让别人一眼看懂代码结构）

我也可以帮你把 README **优化成 JMLR / NeurIPS 项目常见风格**。
