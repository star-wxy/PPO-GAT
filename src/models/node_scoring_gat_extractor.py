import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATConv


class NodeScoringGATFeatureExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        num_nodes: int,
        num_robots: int,
        node_cpu_capacities: list[float],
        node_latencies: list[float],
        node_energy_factors: list[float],
        max_task_size: float,
        max_task_deadline: float,
        max_task_priority: int,
        max_robot_energy: float,
        max_robot_local_cpu: float,
        features_dim: int = 128,
        hidden_dim: int = 32,
        gat_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        self.num_nodes = num_nodes
        self.num_robots = num_robots
        self.total_graph_nodes = 1 + num_nodes + num_robots
        self.node_feat_dim = 17
        self.hidden_dim = hidden_dim
        self.obs_dim = int(observation_space.shape[0])

        cpu_tensor = torch.tensor(node_cpu_capacities, dtype=torch.float32)
        latency_tensor = torch.tensor(node_latencies, dtype=torch.float32)
        energy_tensor = torch.tensor(node_energy_factors, dtype=torch.float32)

        self.register_buffer("node_cpu_capacities", cpu_tensor)
        self.register_buffer("node_latencies", latency_tensor)
        self.register_buffer("node_energy_factors", energy_tensor)
        self.register_buffer("base_edge_index", self._build_edge_index())

        self.max_task_size = max(float(max_task_size), 1.0)
        self.max_task_deadline = max(float(max_task_deadline), 1.0)
        self.max_task_priority = max(float(max_task_priority), 1.0)
        self.max_robot_energy = max(float(max_robot_energy), 1.0)
        self.max_robot_local_cpu = max(float(max_robot_local_cpu), 1.0)
        self.max_node_cpu = max(float(cpu_tensor.max().item()), 1.0)
        self.max_latency = max(float(latency_tensor.max().item()), 1e-3)
        self.max_energy_factor = max(float(energy_tensor.max().item()), 1.0)

        self.input_proj = nn.Sequential(
            nn.Linear(self.node_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gat1 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

        self.raw_encoder = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )

        node_input_dim = hidden_dim * 3 + 7 + 15
        self.node_scorer = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.node_descriptor = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.heuristic_gate = nn.Parameter(torch.tensor(0.6))
        self.score_scale = nn.Parameter(torch.tensor(1.8))

        fusion_input_dim = num_nodes + num_nodes * hidden_dim + hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, features_dim - num_nodes),
            nn.SiLU(),
        )

    def _build_edge_index(self) -> torch.Tensor:
        edges = []
        task_idx = 0
        compute_start = 1
        robot_start = 1 + self.num_nodes

        for i in range(self.num_nodes):
            c_idx = compute_start + i
            edges.append([task_idx, c_idx])
            edges.append([c_idx, task_idx])

        for r in range(self.num_robots):
            r_idx = robot_start + r
            edges.append([task_idx, r_idx])
            edges.append([r_idx, task_idx])

        for r in range(self.num_robots):
            r_idx = robot_start + r
            for i in range(self.num_nodes):
                c_idx = compute_start + i
                edges.append([r_idx, c_idx])
                edges.append([c_idx, r_idx])

        for i in range(self.num_nodes - 1):
            ci = compute_start + i
            cj = compute_start + i + 1
            edges.append([ci, cj])
            edges.append([cj, ci])

        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def _parse_obs(self, observations: torch.Tensor):
        idx = 0
        task_size = observations[:, idx : idx + 1]
        idx += 1
        task_deadline = observations[:, idx : idx + 1]
        idx += 1
        task_priority = observations[:, idx : idx + 1]
        idx += 1
        task_type_norm = observations[:, idx : idx + 1]
        idx += 1
        current_robot_id_norm = observations[:, idx : idx + 1]
        idx += 1
        current_robot_energy = observations[:, idx : idx + 1]
        idx += 1
        current_robot_local_cpu = observations[:, idx : idx + 1]
        idx += 1
        current_robot_queue_norm = observations[:, idx : idx + 1]
        idx += 1
        node_free_cpu = observations[:, idx : idx + self.num_nodes]
        idx += self.num_nodes
        node_latency = observations[:, idx : idx + self.num_nodes]
        idx += self.num_nodes
        node_load_ratio = observations[:, idx : idx + self.num_nodes]
        idx += self.num_nodes
        robot_energy = observations[:, idx : idx + self.num_robots]
        idx += self.num_robots
        robot_local_cpu = observations[:, idx : idx + self.num_robots]
        idx += self.num_robots
        robot_queue_norm = observations[:, idx : idx + self.num_robots]
        return (
            task_size,
            task_deadline,
            task_priority,
            task_type_norm,
            current_robot_id_norm,
            current_robot_energy,
            current_robot_local_cpu,
            current_robot_queue_norm,
            node_free_cpu,
            node_latency,
            node_load_ratio,
            robot_energy,
            robot_local_cpu,
            robot_queue_norm,
        )

    def _normalize_observations(self, observations: torch.Tensor) -> torch.Tensor:
        idx = 0
        obs_chunks = []
        obs_chunks.append(observations[:, idx : idx + 1] / self.max_task_size)
        idx += 1
        obs_chunks.append(observations[:, idx : idx + 1] / self.max_task_deadline)
        idx += 1
        obs_chunks.append(observations[:, idx : idx + 1] / self.max_task_priority)
        idx += 1
        obs_chunks.append(observations[:, idx : idx + 1])
        idx += 1
        obs_chunks.append(observations[:, idx : idx + 1])
        idx += 1
        obs_chunks.append(observations[:, idx : idx + 1] / self.max_robot_energy)
        idx += 1
        obs_chunks.append(observations[:, idx : idx + 1] / self.max_robot_local_cpu)
        idx += 1
        obs_chunks.append(observations[:, idx : idx + 1].clamp(0.0, 3.0))
        idx += 1
        obs_chunks.append(observations[:, idx : idx + self.num_nodes] / self.max_node_cpu)
        idx += self.num_nodes
        obs_chunks.append(observations[:, idx : idx + self.num_nodes] / self.max_latency)
        idx += self.num_nodes
        obs_chunks.append(observations[:, idx : idx + self.num_nodes].clamp(0.0, 1.5))
        idx += self.num_nodes
        obs_chunks.append(observations[:, idx : idx + self.num_robots] / self.max_robot_energy)
        idx += self.num_robots
        obs_chunks.append(observations[:, idx : idx + self.num_robots] / self.max_robot_local_cpu)
        idx += self.num_robots
        obs_chunks.append(observations[:, idx : idx + self.num_robots].clamp(0.0, 3.0))
        return torch.cat(obs_chunks, dim=1)

    def _obs_to_graph(self, observations: torch.Tensor):
        batch_size = observations.shape[0]
        device = observations.device
        (
            task_size,
            task_deadline,
            task_priority,
            task_type_norm,
            current_robot_id_norm,
            current_robot_energy,
            current_robot_local_cpu,
            current_robot_queue_norm,
            node_free_cpu,
            node_latency,
            node_load_ratio,
            robot_energy,
            robot_local_cpu,
            robot_queue_norm,
        ) = self._parse_obs(observations)

        task_size_norm = task_size / self.max_task_size
        task_deadline_norm = task_deadline / self.max_task_deadline
        task_priority_norm = task_priority / self.max_task_priority
        current_robot_id_clamped = current_robot_id_norm.clamp(0.0, 1.0)
        current_robot_energy_norm = current_robot_energy / self.max_robot_energy
        current_robot_local_cpu_norm = current_robot_local_cpu / self.max_robot_local_cpu

        node_capacity = self.node_cpu_capacities.to(device).unsqueeze(0).expand(batch_size, -1)
        node_latency_fixed = self.node_latencies.to(device).unsqueeze(0).expand(batch_size, -1)
        node_energy_factor = self.node_energy_factors.to(device).unsqueeze(0).expand(batch_size, -1)
        node_position = torch.linspace(0.0, 1.0, self.num_nodes, device=device).unsqueeze(0).expand(batch_size, -1)

        node_free_norm = node_free_cpu / self.max_node_cpu
        node_latency_norm = node_latency / self.max_latency
        node_load = node_load_ratio.clamp(0.0, 1.5)
        robot_energy_norm = robot_energy / self.max_robot_energy
        robot_local_cpu_norm = robot_local_cpu / self.max_robot_local_cpu

        avg_free_norm = node_free_norm.mean(dim=1, keepdim=True)
        avg_latency_norm = node_latency_norm.mean(dim=1, keepdim=True)
        avg_load = node_load.mean(dim=1, keepdim=True)
        robot_energy_mean = robot_energy_norm.mean(dim=1, keepdim=True)
        robot_energy_min = robot_energy_norm.min(dim=1, keepdim=True).values
        robot_energy_max = robot_energy_norm.max(dim=1, keepdim=True).values
        robot_queue_mean = robot_queue_norm.mean(dim=1, keepdim=True)

        predicted_compute = task_size / node_capacity.clamp_min(1e-3)
        predicted_queue = node_load * 1.5
        predicted_total = predicted_compute + predicted_queue + node_latency_fixed
        slack = (task_deadline - predicted_total) / self.max_task_deadline
        overload_margin = (node_free_cpu - task_size) / node_capacity.clamp_min(1e-3)
        task_to_capacity = task_size / node_capacity.clamp_min(1e-3)
        task_node = torch.cat(
            [
                task_size_norm,
                task_deadline_norm,
                task_priority_norm,
                task_type_norm,
                avg_free_norm,
                avg_load,
                avg_latency_norm,
                current_robot_id_norm,
                current_robot_energy_norm,
                current_robot_local_cpu_norm,
                current_robot_queue_norm,
                robot_energy_mean,
                robot_queue_mean,
                torch.ones_like(task_size_norm),
                torch.zeros_like(task_size_norm),
                torch.zeros_like(task_size_norm),
                torch.zeros_like(task_size_norm),
            ],
            dim=1,
        ).unsqueeze(1)

        compute_nodes = torch.cat(
            [
                task_size_norm.expand(-1, self.num_nodes).unsqueeze(-1),
                task_deadline_norm.expand(-1, self.num_nodes).unsqueeze(-1),
                task_priority_norm.expand(-1, self.num_nodes).unsqueeze(-1),
                task_type_norm.expand(-1, self.num_nodes).unsqueeze(-1),
                node_free_norm.unsqueeze(-1),
                node_load.unsqueeze(-1),
                node_latency_norm.unsqueeze(-1),
                (node_capacity / self.max_node_cpu).unsqueeze(-1),
                task_to_capacity.unsqueeze(-1),
                slack.unsqueeze(-1),
                current_robot_id_norm.expand(-1, self.num_nodes).unsqueeze(-1),
                current_robot_energy_norm.expand(-1, self.num_nodes).unsqueeze(-1),
                current_robot_local_cpu_norm.expand(-1, self.num_nodes).unsqueeze(-1),
                current_robot_queue_norm.expand(-1, self.num_nodes).unsqueeze(-1),
                torch.ones(batch_size, self.num_nodes, 1, device=device),
                torch.zeros(batch_size, self.num_nodes, 1, device=device),
                (node_energy_factor / self.max_energy_factor).unsqueeze(-1),
            ],
            dim=-1,
        )

        robot_nodes = torch.cat(
            [
                task_size_norm.expand(-1, self.num_robots).unsqueeze(-1),
                task_deadline_norm.expand(-1, self.num_robots).unsqueeze(-1),
                task_priority_norm.expand(-1, self.num_robots).unsqueeze(-1),
                task_type_norm.expand(-1, self.num_robots).unsqueeze(-1),
                avg_free_norm.expand(-1, self.num_robots).unsqueeze(-1),
                avg_load.expand(-1, self.num_robots).unsqueeze(-1),
                avg_latency_norm.expand(-1, self.num_robots).unsqueeze(-1),
                current_robot_id_norm.expand(-1, self.num_robots).unsqueeze(-1),
                robot_energy_norm.unsqueeze(-1),
                robot_local_cpu_norm.unsqueeze(-1),
                robot_queue_norm.unsqueeze(-1),
                current_robot_queue_norm.expand(-1, self.num_robots).unsqueeze(-1),
                (torch.arange(self.num_robots, device=device).view(1, -1, 1) == (
                    current_robot_id_clamped * max(self.num_robots - 1, 1)
                ).round().view(-1, 1, 1)).float(),
                torch.zeros(batch_size, self.num_robots, 1, device=device),
                torch.ones(batch_size, self.num_robots, 1, device=device),
                torch.zeros(batch_size, self.num_robots, 1, device=device),
                torch.zeros(batch_size, self.num_robots, 1, device=device),
            ],
            dim=-1,
        )

        all_nodes = torch.cat([task_node, compute_nodes, robot_nodes], dim=1)
        x = all_nodes.reshape(batch_size * self.total_graph_nodes, self.node_feat_dim)

        edge_indices = []
        base_edge_index = self.base_edge_index.to(device)
        for b in range(batch_size):
            edge_indices.append(base_edge_index + b * self.total_graph_nodes)
        edge_index = torch.cat(edge_indices, dim=1)

        heuristics = torch.stack(
            [
                node_free_norm,
                node_load,
                node_latency_norm,
                task_to_capacity,
                overload_margin,
                predicted_total / self.max_task_deadline,
                slack,
            ],
            dim=-1,
        )

        local_raw = torch.stack(
            [
                node_free_norm,
                node_load,
                node_latency_norm,
                (node_capacity / self.max_node_cpu),
                (node_energy_factor / self.max_energy_factor),
                task_size_norm.expand(-1, self.num_nodes),
                task_deadline_norm.expand(-1, self.num_nodes),
                task_priority_norm.expand(-1, self.num_nodes),
                task_type_norm.expand(-1, self.num_nodes),
                current_robot_id_clamped.expand(-1, self.num_nodes),
                current_robot_energy_norm.expand(-1, self.num_nodes),
                current_robot_local_cpu_norm.expand(-1, self.num_nodes),
                current_robot_queue_norm.expand(-1, self.num_nodes),
                task_to_capacity,
                overload_margin,
            ],
            dim=-1,
        )
        return x, edge_index, heuristics, local_raw

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x, edge_index, heuristics, local_raw = self._obs_to_graph(observations)

        x = self.input_proj(x)

        residual = x
        x = self.gat1(x, edge_index)
        x = self.norm1(residual + self.dropout(x))

        residual = x
        x = self.gat2(x, edge_index)
        x = self.norm2(residual + self.dropout(x))

        residual = x
        x = residual + self.dropout(self.ffn(x))
        x = x.view(batch_size, self.total_graph_nodes, self.hidden_dim)

        task_embed = x[:, 0]
        compute_embeds = x[:, 1 : 1 + self.num_nodes]
        robot_embeds = x[:, 1 + self.num_nodes :]
        robot_mean = robot_embeds.mean(dim=1) if robot_embeds.shape[1] > 0 else torch.zeros_like(task_embed)
        global_mean = x.mean(dim=1)

        node_inputs = torch.cat(
            [
                compute_embeds,
                task_embed.unsqueeze(1).expand(-1, self.num_nodes, -1),
                robot_mean.unsqueeze(1).expand(-1, self.num_nodes, -1),
                heuristics,
                local_raw,
            ],
            dim=-1,
        )

        learned_scores = self.node_scorer(node_inputs).squeeze(-1)
        node_free = heuristics[..., 0]
        node_load = heuristics[..., 1]
        node_latency = heuristics[..., 2]
        task_to_capacity = heuristics[..., 3]
        overload_margin = heuristics[..., 4]
        slack = heuristics[..., 6]
        current_robot_pos = local_raw[..., 9]
        current_robot_queue = local_raw[..., 12]
        task_deadline_norm = local_raw[..., 6]
        energy_factor = local_raw[..., 4]
        local_cpu_signal = local_raw[..., 11]
        node_position = torch.linspace(
            0.0,
            1.0,
            self.num_nodes,
            device=observations.device,
        ).unsqueeze(0).expand(batch_size, -1)
        distance_norm = (node_position - current_robot_pos).abs()
        queue_pressure = node_load * (1.0 + task_to_capacity)
        overload_risk = torch.relu(-overload_margin)
        deadline_risk = torch.relu(-slack)
        energy_proxy = energy_factor * (0.55 + 0.45 * local_raw[..., 5])
        backlog_pressure = current_robot_queue * (1.0 + 0.6 * deadline_risk)
        remote_cost = distance_norm * (0.8 + 0.6 * current_robot_queue + 0.4 * deadline_risk)
        near_edge_bonus = (1.0 - distance_norm) * (0.45 + 0.35 * local_cpu_signal)
        cloud_risk = node_latency * energy_factor * (0.9 + 0.8 * deadline_risk + 0.5 * backlog_pressure)
        service_headroom = node_free - 0.55 * task_to_capacity - 0.65 * node_load
        cloud_mask = (node_latency > 0.72).float()
        edge_mask = 1.0 - cloud_mask
        edge_load_mean = (node_load * edge_mask).sum(dim=1, keepdim=True) / edge_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        edge_headroom_mean = (service_headroom * edge_mask).sum(dim=1, keepdim=True) / edge_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        edge_congestion = torch.relu(edge_load_mean - 0.42)
        edge_shortage = torch.relu(0.12 - edge_headroom_mean)
        cloud_enable = cloud_mask * (edge_congestion + 0.8 * edge_shortage)
        transmission_friendly = torch.relu(0.95 - task_to_capacity)
        relaxed_deadline = torch.relu(task_deadline_norm - 0.55)
        cloud_relief_bonus = cloud_enable * (
            1.2 * transmission_friendly
            + 0.9 * relaxed_deadline
            + 0.7 * torch.relu(node_free - task_to_capacity)
            + 0.45 * backlog_pressure
        )
        cloud_guard_penalty = cloud_mask * (
            1.35 * torch.relu(0.18 - edge_congestion)
            + 0.9 * torch.relu(task_to_capacity - 0.95)
            + 0.75 * torch.relu(0.52 - task_deadline_norm)
        )

        heuristic_scores = (
            1.9 * slack
            + 1.35 * overload_margin
            + 1.25 * service_headroom
            + 0.65 * near_edge_bonus
            + 0.95 * cloud_relief_bonus
            - 1.85 * queue_pressure
            - 1.55 * overload_risk
            - 1.45 * deadline_risk
            - 1.05 * task_to_capacity
            - 0.95 * remote_cost
            - 0.72 * node_latency
            - 0.32 * energy_proxy
            - 0.85 * backlog_pressure
            - 0.9 * cloud_risk
            - 1.1 * cloud_guard_penalty
        )
        gate = self.heuristic_gate.clamp(0.0, 1.0)
        node_scores = self.score_scale.clamp(0.8, 3.0) * (
            gate * heuristic_scores + (1.0 - gate) * learned_scores
        )
        node_descriptors = self.node_descriptor(node_inputs).reshape(batch_size, -1)
        raw_features = self.raw_encoder(self._normalize_observations(observations))

        fused_context = self.fusion(
            torch.cat([node_scores, node_descriptors, raw_features, task_embed, global_mean], dim=1)
        )
        return torch.cat([node_scores, fused_context], dim=1)
