import inspect
import functools

def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    if args.small_env:
        from .simple_push2_small import parallel_env
        print("making a small env")
    else:
        from .simple_push2 import parallel_env
    
    env = parallel_env(max_cycles=args.max_episode_len, continuous_actions=False, render_mode=args.render_mode)
    env.reset()

    if not args.lever_cue and args.lever_action:
        raise Exception("can't do lever action without lever cue. because I said so")

    '''if args.lever_cue is not None:
        args.obs_shape = [14, 14] 
    else: 
        args.obs_shape = [13, 13]
    if args.lever_action:
        args.obs_shape = [x + 2 for x in args.obs_shape]  #NEW_CODE increased observation space by +2 instead of +1'''
    
    if args.lever_cue is not None:
        args.obs_shape = [7, 7] 
    else: 
        args.obs_shape = [6, 6]
    if args.lever_action:
        args.obs_shape = [x + 2 for x in args.obs_shape]  #NEW_CODE increased observation space by +2 instead of +1
    
            
    args.action_shape = [args.num_actions, args.num_actions]
    args.action_shape = [x + 2 for x in args.action_shape] # this is because we have to gaze / not gaze right now
    
    if args.lever_action:
        args.action_shape = [x + 1 for x in args.action_shape]
        args.num_actions += 1 #Add Lever Action

    args.n_players = 2
    args.n_agents = 2
    args.high_action = 1
    args.low_action = -1
    return env, args
