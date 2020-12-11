# Game of Drones Environment (How to use it)

*  **Initializing the environment:**
    
        from game_of_drones import GameOfDrone

        env = GameOfDrone()
                  

*  **Resetting the state space of the environment:**
    
        state_space = env.reset()

    (*state_space* is an Object of *States* class in `states_drone.py` and will return the relevant information about the drone's dynamics, i.e. the distance of the drone from the target, its current velocity, angle and angular velocity)

*  **Taking a step in the environment:**
    
        state_space, reward, done = env.step(action_value) 
    *action_value* is an object of *Actions* class in `states_drone.py`

# A closer look into the state space

The state space will return 6 drone's property. The first two correspond to the distance between the drone's body and the target in spatial coordinates. The second two correspond to the xy coordinate of the current velocity vector and the last two correspond to the angle and angular velocity of the drone. 
All the properties are normalized, specifically the spatial coordinates of the distance are within 0 and 1, the velocity components are between -1 and 1, the angle is between 0 and 1 and the angular velocity is between -1 and 1


# Setting environment complexities e.g. wind and controller disturbance

There is a flag *controller_disturbance* inside *init()* method of *GameofDrone* class which can be set to *True* which would lead to approximately 10% chance of the environment not taking the desired action, but instead a random one.
  
    

    env.controller_disturbance = True

There is a flag *wind_disturbance* inside *init()* method of *GameofDrone* class which can be set to *True* which would lead to altering the position of the drone. The direction from which the wind will blow i.e. north, south, east or west and its power are set randomly at the start of each episode.
  
    

    env.wind_disturbance = True
    
# Sample way to discretize the state space

There is a sample method *discretize()* in the *States* class in `states_drone.py` which takes the state space as input parameter and returns a discretized 3-dimensional state space: Example code on how to use it has been provided below:
  
    

    from states_drone import States

    discrete = state_space.discretize()

# Using PID controller to stabilze the drone horizontally in the air (sample use)

In the bottom section of the 'game_of_drones.py', we show how to employ the PID for a simple test case. 
The linear horizontal velocity of the drone requires to be set to 0 otherwise it will increase gradually due to unexpected swinging behavior due to turbulances.


*  There is a file called `PID.py` in which *PID* class can be initialized with parameters of *P, I & D*
    
        from PID import PID

        pid = PID(0.5, 0.5, 0.5)

*  Stabilze the drone 
        
        action = pid.pid_control(states)
        env.drone.linearVelocity = [0, env.drone.linearVelocity[1]]
        print("PID controller's chosen action: ", action)
        
