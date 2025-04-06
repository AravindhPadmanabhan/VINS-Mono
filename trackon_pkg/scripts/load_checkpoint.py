import os
import torch

def restart_from_checkpoint_not_dist(args, run_variables, **kwargs):

    checkpoint_path = args.checkpoint_path

    assert checkpoint_path is not None
    assert os.path.exists(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model"].items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v

    checkpoint["model"] = new_state_dict

    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint with msg {}".format(key, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint".format(key))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint".format(key))
        else:
            print("=> key '{}' not found in checkpoint".format(key))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]