from typing import List, Tuple

import pytorch_lightning as pl
from torch.optim import Adam

from .networks import GATPolicyNetwork, GCNPolicyNetwork, SAGEPolicyNetwork
from .data import ExperienceSourceDataset
from .lib import Language, EGraph
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import gym
import torch_geometric as geom
from .ReplayBuffer import *
from collections import OrderedDict, deque, namedtuple
from typing import List, Tuple
from torch import Tensor, nn
from pytorch_lightning import LightningModule, Trainer
from torch.optim import Adam, Optimizer


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
            # print("random", action)
        else:
            # state = torch.tensor([self.state])

            # if device not in ["cpu"]:
                # state = state.cuda(device)
            q_values = net(self.state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
        is_terminal=False
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)
        done = True if is_terminal else done

        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
        self,
        lang,
        expr,
        batch_size: int = 16,
        lr: float = 1e-2,
        env: str = "CartPole-v0",
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 20,
        warm_start_steps: int = 1000,
    ) -> None:
        """
        Args:
            batch_size: size of the batches")
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters()

        self.env = gym.make(self.hparams.env, lang=lang, expr=expr)
        obs_size = self.env.num_node_features
        n_actions = self.env.action_space.n

        self.net = SAGEPolicyNetwork(obs_size, n_actions)
        self.target_net = SAGEPolicyNetwork(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.episode_steps = 0
        self.populate(self.hparams.warm_start_steps)
        self.env.reset()

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially fill up the replay buffer with
        experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)
        print("finished populating")

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch
        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        state_action_values = state_action_values.to(torch.float64)
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards
        loss = nn.MSELoss()(state_action_values, expected_state_action_values)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
        """Carries out a single step through the environment to update the replay buffer. Then calculates loss
        based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        self.episode_steps += 1
        device = self.get_device(batch)
        epsilon = self.hparams.eps_start
        # epsilon = max(
        #     self.hparams.eps_end,
        #     self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame,
        # )

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device,is_terminal=self.episode_steps == self.hparams.episode_length)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        # if self.trainer._distrib_type in {DistributedType.DP, DistributedType.DDP2}:
        #     loss = loss.unsqueeze(0)
        if done:
            self.total_reward = self.episode_reward
            self.log("episode_reward", torch.tensor(self.episode_reward).to(device), prog_bar=True, on_step=False, on_epoch=True, batch_size=self.hparams.batch_size)
            self.episode_reward = 0
            self.episode_steps = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device).to(torch.float32),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }

        # self.log("total_reward", log["total_reward"], batch_size=self.hparams.batch_size)
        # self.log("reward", log["reward"], batch_size=self.hparams.batch_size, on_step=True),

        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=exp_collate,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"