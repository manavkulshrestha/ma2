import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, *, num_humans, num_robots, num_goals):
        world = World()
        # set any world properties first
        world.dim_c = 2

        # add agents (robot + human)
        world.robots = [Agent() for i in range(num_robots)]
        world.humans = [Agent() for i in range(num_humans)]
        for i, agent in enumerate(world.robots):
            agent.name = f'robot{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.color = np.array([0.75, 0.25, 0.25])
        for i, agent in enumerate(world.humans):
            agent.name = f'human{i}'
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.color = np.array([0.75, 0.75, 0.25])
        world.agents = world.humans+world.robots
        world.robots_eprev = np.zeros([len(world.robots), 2])
        world.humans_eprev = np.zeros([len(world.humans), 2])

        # add goals
        world.landmarks = [Landmark() for i in range(num_goals)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f'goal{i}'
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.2
            landmark.color = np.array([0.25, 0.75, 0.25])

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for landmark in world.landmarks:
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # # returns data for benchmarking purposes
        # if agent.adversary:
        #     return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        # else:
        #     dists = []
        #     for l in world.landmarks:
        #         dists.append(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
        #     dists.append(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
        #     return tuple(dists)
        return ()

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        # return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return 0

    def observation(self, agent, world):
        # # get positions of all entities in this agent's reference frame
        # entity_pos = []
        # for entity in world.landmarks:
        #     entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # # entity colors
        # entity_color = []
        # for entity in world.landmarks:
        #     entity_color.append(entity.color)
        # # communication of all other agents
        # other_pos = []
        # for other in world.agents:
        #     if other is agent: continue
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)

        # if not agent.adversary:
        #     return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
        # else:
        #     return np.concatenate(entity_pos + other_pos)
        return [0]*len(world.agents)