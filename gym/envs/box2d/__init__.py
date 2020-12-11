try:
    import Box2D
    from gym.envs.box2d.lunar_lander import LunarLander
    from gym.envs.box2d.lunar_lander import LunarLanderContinuous
except ImportError:
    Box2D = None
