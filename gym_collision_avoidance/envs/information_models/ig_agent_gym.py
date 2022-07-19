import numpy as np

from gym_collision_avoidance.envs.information_models.ig_agent import ig_agent


class IG_agent_gym(ig_agent):
    def __init__(self, host_agent, expert_policy=None):
        super().__init__(expert_policy)

        self.host_agent = host_agent

        self.global_pose = np.array([0., 0.])

    def _init_model(self, **kwargs):
        pass

    def update(self, agents, num_steps):

        pos_global_frame = self.host_agent.get_agent_data("pos_global_frame")
        heading_global_frame = self.host_agent.get_agent_data("heading_global_frame")

        self.global_pose = np.append(pos_global_frame, heading_global_frame)

        self._update_belief(agents)

        self.update_agent_pos_map()

        self.finished = self.targetMap.finished

    def _update_belief(self, agents):
        targets = []
        poses = []
        # Find Targets in Range and FOV (Detector Emulation)
        self.obsvd_targets = self._find_targets_in_obs(agents, self.global_pose)
        targets.append(self.obsvd_targets)
        poses.append(self.global_pose)
        # Get observations of other agens
        ig_agents = [
            i
            for i in range(len(agents))
            if "ig_" in str(type(agents[i].policy)) and i != self.host_agent.id
        ]
        for j in ig_agents:
            other_agent_targets = (
                agents[j].policy.obsvd_targets
                if agents[j].policy.obsvd_targets is not None
                else []
            )
            targets.append(other_agent_targets)
            other_agent_pose = np.append(
                agents[j].pos_global_frame, agents[j].heading_global_frame
            )
            poses.append(other_agent_pose)

        # Update Target Map
        self.team_obsv_cells, self.team_reward = self.targetMap.update(
            poses, targets, frame="global"
        )
        # self.team_reward = self.targetMap.get_reward_from_cells(self.team_obsv_cells)

    def _find_targets_in_obs(self, agents, global_pose):

        # Find Targets in Range and FOV (Detector Emulation)
        c, s = np.cos(global_pose[2]), np.sin(global_pose[2])
        R = np.array(((c, s), (-s, c)))
        targets = []
        for agent in agents:
            if agent.policy.str == "StaticPolicy":
                # Static Agent = Target
                r = agent.pos_global_frame - global_pose[0:2]
                r_norm = np.sqrt(r[0] ** 2 + r[1] ** 2)
                in_range = r_norm <= self.detect_range
                if in_range:
                    r_rot = np.dot(R, r)
                    dphi = (np.arctan2(r_rot[1], r_rot[0]))
                    in_fov = abs(dphi) <= self.detect_fov / 2.0
                    if in_fov:
                        targets.append(agent.pos_global_frame)
                    else:
                        what = 1

        return targets
