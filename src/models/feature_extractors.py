import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv


class GATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        num_nodes: int,
        num_robots: int,
        features_dim: int = 64,
        hidden_dim: int = 32,
        gat_heads: int = 4,
    ):
        super().__init__(observation_space, features_dim)

        self.num_nodes = num_nodes
        self.num_robots = num_robots

        # 1 task + N compute + R robot
        self.total_graph_nodes = 1 + num_nodes + num_robots
        self.hidden_dim = hidden_dim

        # richer node features
        self.node_feat_dim = 8

        self.input_proj = nn.Sequential(
            nn.Linear(self.node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gat1 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=False,
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=False,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        fusion_input_dim = hidden_dim * (self.num_nodes + 3)
        self.raw_encoder = nn.Sequential(
            nn.Linear(int(observation_space.shape[0]), hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, features_dim),
            nn.ReLU(),
        )

        self._edge_index = self._build_edge_index()

    def _build_edge_index(self) -> torch.Tensor:
        """
        node ids:
        0 = task
        1..N = compute nodes
        1+N .. 1+N+R-1 = robot nodes
        """
        edges = []

        task_idx = 0
        compute_start = 1
        robot_start = 1 + self.num_nodes

        # 1) task <-> compute
        for i in range(self.num_nodes):
            c_idx = compute_start + i
            edges.append([task_idx, c_idx])
            edges.append([c_idx, task_idx])

        # 2) task <-> robots
        for r in range(self.num_robots):
            r_idx = robot_start + r
            edges.append([task_idx, r_idx])
            edges.append([r_idx, task_idx])

        # 3) robot <-> compute
        for r in range(self.num_robots):
            r_idx = robot_start + r
            for i in range(self.num_nodes):
                c_idx = compute_start + i
                edges.append([r_idx, c_idx])
                edges.append([c_idx, r_idx])

        # 4) sparse compute topology
        for i in range(self.num_nodes - 1):
            ci = compute_start + i
            cj = compute_start + i + 1
            edges.append([ci, cj])
            edges.append([cj, ci])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def _obs_to_graph(self, observations: torch.Tensor):
        """
        Supports both the legacy observation layout and the current layout
        with extra current-robot fields after task priority.
        """
        batch_size = observations.shape[0]
        device = observations.device
        obs_dim = observations.shape[1]
        new_obs_dim = 5 + self.num_nodes * 3 + self.num_robots

        idx = 0
        task_size = observations[:, idx : idx + 1]
        idx += 1

        task_deadline = observations[:, idx : idx + 1]
        idx += 1

        task_priority = observations[:, idx : idx + 1]
        idx += 1

        current_robot_id = None
        current_robot_energy = None
        if obs_dim == new_obs_dim:
            current_robot_id = observations[:, idx : idx + 1]
            idx += 1
            current_robot_energy = observations[:, idx : idx + 1]
            idx += 1

        node_free_cpu = observations[:, idx : idx + self.num_nodes]
        idx += self.num_nodes

        node_latency = observations[:, idx : idx + self.num_nodes]
        idx += self.num_nodes

        node_load_ratio = observations[:, idx : idx + self.num_nodes]
        idx += self.num_nodes

        robot_energy = observations[:, idx : idx + self.num_robots]
        idx += self.num_robots

        graph_node_features = []
        batch_index = []

        for b in range(batch_size):
            avg_free = node_free_cpu[b].mean().item()
            avg_latency = node_latency[b].mean().item()
            avg_load = node_load_ratio[b].mean().item()
            current_robot_id_value = (
                current_robot_id[b, 0].item() if current_robot_id is not None else 0.0
            )

            # 当前任务默认对应当前活跃机器人，先取第一个机器人能量的近似
            if current_robot_energy is not None:
                current_robot_energy_value = current_robot_energy[b, 0].item()
            else:
                current_robot_energy_value = robot_energy[b, 0].item()

            node_features = []

            # ---- task node ----
            task_node = torch.tensor(
                [
                    task_size[b, 0].item(),
                    task_deadline[b, 0].item(),
                    task_priority[b, 0].item(),
                    current_robot_energy_value,
                    current_robot_id_value,
                    avg_free,
                    avg_latency,
                    avg_load,
                ],
                dtype=torch.float32,
                device=device,
            )
            node_features.append(task_node)

            # ---- compute nodes ----
            for i in range(self.num_nodes):
                # 用 index 近似节点类型编码：
                # local/edge/cloud 暂时不能直接从obs得到，这里先用 latency+free_cpu等特征表达
                c_node = torch.tensor(
                    [
                        node_free_cpu[b, i].item(),            # free cpu
                        node_load_ratio[b, i].item(),
                        node_latency[b, i].item(),
                        task_size[b, 0].item(),
                        task_deadline[b, 0].item(),
                        current_robot_energy_value,
                        current_robot_id_value,
                        i / max(self.num_nodes - 1, 1),
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                node_features.append(c_node)

            # ---- robot nodes ----
            for r in range(self.num_robots):
                r_node = torch.tensor(
                    [
                        robot_energy[b, r].item(),
                        task_size[b, 0].item(),
                        task_deadline[b, 0].item(),
                        task_priority[b, 0].item(),
                        avg_free,
                        avg_latency,
                        avg_load,
                        1.0 if r == int(round(current_robot_id_value * max(self.num_robots - 1, 1))) else 0.0,
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                node_features.append(r_node)

            node_features = torch.stack(node_features, dim=0)
            graph_node_features.append(node_features)
            batch_index.extend([b] * self.total_graph_nodes)

        x = torch.cat(graph_node_features, dim=0)
        batch_index = torch.tensor(batch_index, dtype=torch.long, device=device)

        edge_indices = []
        base_edge_index = self._edge_index.to(device)

        for b in range(batch_size):
            offset = b * self.total_graph_nodes
            edge_indices.append(base_edge_index + offset)

        edge_index = torch.cat(edge_indices, dim=1)

        return x, edge_index, batch_index

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x, edge_index, _ = self._obs_to_graph(observations)

        x = self.input_proj(x)

        residual = x
        x = self.gat1(x, edge_index)
        x = self.norm1(residual + torch.relu(x))

        residual = x
        x = self.gat2(x, edge_index)
        x = self.norm2(residual + torch.relu(x))

        x = x.view(batch_size, self.total_graph_nodes, self.hidden_dim)
        task_embed = x[:, 0]
        compute_embeds = x[:, 1 : 1 + self.num_nodes].reshape(batch_size, -1)
        robot_embeds = x[:, 1 + self.num_nodes :]
        robot_mean = (
            robot_embeds.mean(dim=1)
            if robot_embeds.shape[1] > 0
            else torch.zeros_like(task_embed)
        )
        raw_features = self.raw_encoder(observations)

        fused = torch.cat([task_embed, compute_embeds, robot_mean, raw_features], dim=1)
        return self.mlp(fused)
