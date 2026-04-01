import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv, global_mean_pool


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

        # 节点总数 = 1个任务节点 + N个算力节点 + R个机器人节点
        self.total_graph_nodes = 1 + num_nodes + num_robots

        # 所有节点统一映射到相同维度
        # 节点特征统一设置为6维：
        # [f1, f2, f3, f4, f5, node_type_id]
        self.node_feat_dim = 6

        self.gat1 = GATConv(
            in_channels=self.node_feat_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=True,
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim * gat_heads,
            out_channels=hidden_dim,
            heads=1,
            concat=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

        self._edge_index = self._build_edge_index()

    def _build_edge_index(self) -> torch.Tensor:
        """
        节点编号约定：
        0 = task node
        1..num_nodes = compute nodes
        1+num_nodes .. 1+num_nodes+num_robots-1 = robot nodes
        """
        edges = []

        task_idx = 0
        compute_start = 1
        robot_start = 1 + self.num_nodes

        # task <-> each compute node
        for i in range(self.num_nodes):
            c_idx = compute_start + i
            edges.append([task_idx, c_idx])
            edges.append([c_idx, task_idx])

        # each robot <-> each compute node
        for r in range(self.num_robots):
            r_idx = robot_start + r
            for i in range(self.num_nodes):
                c_idx = compute_start + i
                edges.append([r_idx, c_idx])
                edges.append([c_idx, r_idx])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def _obs_to_graph(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        observations shape: [batch_size, obs_dim]
        输出：
        x: 所有图节点拼接后的特征 [batch_size * total_graph_nodes, node_feat_dim]
        edge_index: [2, total_edges]
        batch_index: [batch_size * total_graph_nodes]
        """
        batch_size = observations.shape[0]
        device = observations.device

        # observation结构：
        # [task_size, task_deadline, task_priority,
        #  node_free_cpu * N,
        #  node_latency * N,
        #  node_load_ratio * N,
        #  robot_energy * R]

        idx = 0
        task_size = observations[:, idx : idx + 1]
        idx += 1

        task_deadline = observations[:, idx : idx + 1]
        idx += 1

        task_priority = observations[:, idx : idx + 1]
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
            node_features = []

            # 1. task node
            # [task_size, deadline, priority, 0, 0, node_type_id]
            task_node = torch.tensor(
                [
                    task_size[b, 0].item(),
                    task_deadline[b, 0].item(),
                    task_priority[b, 0].item(),
                    0.0,
                    0.0,
                    0.0,  # task type
                ],
                dtype=torch.float32,
                device=device,
            )
            node_features.append(task_node)

            # 2. compute nodes
            for i in range(self.num_nodes):
                c_node = torch.tensor(
                    [
                        node_free_cpu[b, i].item(),
                        node_latency[b, i].item(),
                        node_load_ratio[b, i].item(),
                        0.0,
                        0.0,
                        1.0,  # compute type
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                node_features.append(c_node)

            # 3. robot nodes
            for r in range(self.num_robots):
                r_node = torch.tensor(
                    [
                        robot_energy[b, r].item(),
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        2.0,  # robot type
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

        # 为batch中的每张图复制edge_index，并做节点偏移
        edge_indices = []
        base_edge_index = self._edge_index.to(device)

        for b in range(batch_size):
            offset = b * self.total_graph_nodes
            edge_indices.append(base_edge_index + offset)

        edge_index = torch.cat(edge_indices, dim=1)

        return x, edge_index, batch_index

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x, edge_index, batch_index = self._obs_to_graph(observations)

        x = self.gat1(x, edge_index)
        x = torch.relu(x)

        x = self.gat2(x, edge_index)
        x = torch.relu(x)

        # 图级别池化
        x = global_mean_pool(x, batch_index)

        # 输出给 PPO 的特征
        x = self.mlp(x)
        return x