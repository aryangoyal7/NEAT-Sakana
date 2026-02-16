from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
import jax.numpy as jnp


ActionFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class EpisodeResult:
    right_reward: float
    left_reward: float
    steps: int
    frames: list[Image.Image]


class SlimeVolleyEvaluator:
    def __init__(self, max_steps: int = 3000, frame_skip: int = 4):
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.sv = self._import_slimevolley()

    @staticmethod
    def _import_slimevolley():
        try:
            from evojax.task import slimevolley as sv
        except Exception as exc:  # pragma: no cover - import error path
            raise RuntimeError(
                "Could not import evojax.task.slimevolley. Ensure local EvoJAX is in PYTHONPATH."
            ) from exc
        return sv

    def _sample_ball_velocity(self, rng: np.random.Generator) -> tuple[float, float]:
        rnd = rng.uniform(-1.0, 1.0, size=(2,))
        return float(rnd[0] * 20.0), float(rnd[1] * 7.5 + 17.5)

    def _new_game(self, rng: np.random.Generator):
        ball_vx, ball_vy = self._sample_ball_velocity(rng)
        game_state = self.sv.initGameState(ball_vx, ball_vy)
        return self.sv.Game(game_state)

    def _respawn_ball(self, game, rng: np.random.Generator) -> None:
        gs = game.getGameState()
        ball_vx, ball_vy = self._sample_ball_velocity(rng)
        new_ball = self.sv.initParticleState(0, self.sv.REF_W / 4, ball_vx, ball_vy, 0.5)
        gs2 = self.sv.GameState(
            ball=new_ball,
            agent_left=gs.agent_left,
            agent_right=gs.agent_right,
            hidden_left=gs.hidden_left,
            hidden_right=gs.hidden_right,
            action_left_flag=gs.action_left_flag,
            action_left=gs.action_left,
            action_right_flag=gs.action_right_flag,
            action_right=gs.action_right,
        )
        game.setGameState(gs2)

    def play(
        self,
        rng: np.random.Generator,
        right_policy: ActionFn,
        left_policy: ActionFn | None = None,
        capture_frames: bool = False,
        early_terminate: bool = True,
    ) -> EpisodeResult:
        game = self._new_game(rng)

        frames: list[Image.Image] = []
        if capture_frames:
            frames.append(Image.fromarray(game.display()))

        right_reward = 0.0
        step = 0

        for step in range(self.max_steps):
            obs_left = np.asarray(game.agent_left.getObservation(), dtype=np.float32)
            obs_right = np.asarray(game.agent_right.getObservation(), dtype=np.float32)

            if left_policy is not None:
                left_action = np.asarray(left_policy(obs_left), dtype=np.float32).reshape(-1)
                game.setLeftAction(jnp.asarray(left_action, dtype=jnp.float32))

            right_action = np.asarray(right_policy(obs_right), dtype=np.float32).reshape(-1)
            game.setRightAction(jnp.asarray(right_action, dtype=jnp.float32))

            game.setAction()
            reward = float(game.step())
            right_reward += reward

            if reward != 0.0:
                self._respawn_ball(game, rng)

            if capture_frames and step % self.frame_skip == 0:
                frames.append(Image.fromarray(game.display()))

            if early_terminate:
                left_life = int(game.agent_left.p.life)
                right_life = int(game.agent_right.p.life)
                if left_life <= 0 or right_life <= 0:
                    break

        return EpisodeResult(
            right_reward=right_reward,
            left_reward=-right_reward,
            steps=step + 1,
            frames=frames,
        )



def save_gif(frames: list[Image.Image], path: Path, duration_ms: int = 40) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
