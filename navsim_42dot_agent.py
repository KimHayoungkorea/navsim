"""42dot VLA-based agent for NAVSIM benchmark evaluation."""

import torch
import numpy as np
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from dataclasses import dataclass


@dataclass
class FortyTwoDotAgentConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    horizon: int = 8        # 4s at 2Hz
    lr: float = 1e-4


class FortyTwoDotNavSimAgent(AbstractAgent):
    """VLA-style agent combining camera features with navigation for NAVSIM."""

    def __init__(self, config: FortyTwoDotAgentConfig = FortyTwoDotAgentConfig()):
        self.config = config
        self._model = self._build_model()

    def _build_model(self) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(1024, self.config.d_model),
            torch.nn.ReLU(),
            *[torch.nn.TransformerEncoderLayer(
                self.config.d_model, self.config.n_heads, batch_first=True
            ) for _ in range(self.config.n_layers)],
            torch.nn.Linear(self.config.d_model, self.config.horizon * 3),  # x, y, heading
        )

    @property
    def sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_lidar_config()

    def get_sensor_config(self) -> SensorConfig:
        return self.sensor_config

    def initialize(self) -> None:
        self._model.eval()

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        with torch.no_grad():
            # Flatten camera features as placeholder (replace with real encoder)
            cam_feat = torch.zeros(1, 1024)
            raw = self._model(cam_feat).squeeze(0)  # (horizon*3,)
            waypoints = raw.view(self.config.horizon, 3).numpy()
        return Trajectory(poses=waypoints)
