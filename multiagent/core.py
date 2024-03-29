import numpy as np
from numpy import cos, sin, pi as PI


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

        # travelled distance
        self.travelled = 0

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.05
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        self.humans = []
        self.robots = []

        self.repulsive_magnitude = 0.1
        self.repulsive_range = 0.3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self, apply_forces=True):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # apply repulsive field forces
        p_force = self.apply_repulsive_force(p_force) # MINE
        # integrate physical state
        if apply_forces:
            self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise 
        # print(np.array(p_force[:-2]))
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force
    
    def apply_repulsive_force(self, p_force):
        robot_locs = np.array([x.state.p_pos for x in self.robots])
        human_locs = np.array([x.state.p_pos for x in self.humans])

        for i, (human, pos) in enumerate(zip(self.humans, human_locs)):
            away_vecs = pos - robot_locs

            # close robots repell
            force = np.array([0.0, 0.0])
            close_robots = np.linalg.norm(away_vecs, axis=-1) < self.repulsive_range

            # if any robot is close, repell from them. Else, random/no movement/move towards center
            if np.any(close_robots):
                for away_vec in away_vecs[close_robots]:
                    magnitude = np.linalg.norm(away_vec)
                    away_unit = away_vec/magnitude
                    force += self.repulsive_magnitude * away_unit/(magnitude**2)
            else:
                # ROBOTS MOVE TOWARDS CENTER
                # towards_c = -pos/np.linalg.norm(pos)
                # force = 0.1 * towards_c
            
                # HUMANS HAVE ACC SET AS PER ROBOT ACT DISTRIBUTION
                # acts mags mean = 0.19390881508547342, std = 0.08747601005810045
                # acts x distr (m=5.834367976106885e-07,s=0.1503169688657627), y distr (m=2.0513113491896712e-05,s=0.15052404908657874)
                # th = np.random.uniform(-np.pi, np.pi)
                # rvec = np.array([np.cos(th), np.sin(th)])
                # rmag = max(0, np.random.normal(loc=0.2, scale=0.09))
                # force = rmag*rvec
                
                # HUMANS HAVE VEL SET AS PER ROBOT VEL DISTRIBUTION
                # vels mags mean = 0.7225523392939881, std = 0.25179845006687984. Leans forward a bit
                # vels x distr (m=0.00033287382472207, s=0.5208129880552299), y distr (m=0.0002948546861350656, s=0.5206568839728843), SEMI-CIRCLE-ISH distributions
                # th = np.random.uniform(-np.pi, np.pi)
                # rvec = np.array([np.cos(th), np.sin(th)])
                # rmag = max(0, np.random.normal(loc=0.75, scale=0.25))
                # human.state.p_vel = rmag*rvec
            
                # HUMANS HAVE SPLINE MOVEMENT unless being herded
                pass

            p_force[i] += force

        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            
            displacement = entity.state.p_vel * self.dt
            entity.state.p_pos += displacement
            entity.state.travelled += np.linalg.norm(displacement)


    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]