import gymnasium as gym
import numpy as np
import mujoco

class InvertedPendulumEnv(gym.Env):
    def __init__(self, urdf_path="inverted_pendulum.urdf", render=False):
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render
        self.viewer = None
        if render:
            self.viewer = self._create_viewer()

        # 观测空间：关节角度、速度
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        # 动作空间：施加力矩
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        mujoco.mj_resetData(self.model, self.data)
        # 初始角度加点随机噪声
        self.data.qpos[:] = np.random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        self.data.qvel[:] = np.random.uniform(low=-0.1, high=0.1, size=self.model.nv)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        return np.array([self.data.qpos[0], self.data.qvel[0]], dtype=np.float32)

    def step(self, action):
        # 施加控制
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        angle = obs[0]

        # 奖励函数：越接近竖直（角度=0）奖励越高
        reward = 1.0 - (angle**2)

        # 终止条件：角度过大（倒了）
        terminated = abs(angle) > np.pi / 2
        truncated = False
        return obs, reward, terminated, truncated, {}

    def close(self):
        if self.viewer is not None:
            close = getattr(self.viewer, "close", None)
            if callable(close):
                close()

    def _create_viewer(self):
        """Create a viewer compatible with different MuJoCo distributions."""
        if hasattr(mujoco, "viewer"):
            return mujoco.viewer.launch_passive(self.model, self.data)

        try:
            import mujoco_viewer
        except ImportError as exc:  # pragma: no cover - dependent on environment
            raise RuntimeError(
                "MuJoCo viewer support is not available. Install `mujoco`>=2.3.4 "
                "or add the optional `mujoco_viewer` dependency."
            ) from exc

        return mujoco_viewer.MujocoViewer(self.model, self.data)

