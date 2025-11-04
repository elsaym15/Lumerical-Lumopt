import warnings
import functools

def beta_feature_required(required_flag_name=None, feature_name=""):
    """Decorator to mark a class as beta and require a flag or argument."""
    def decorator(init):
        @functools.wraps(init)
        def wrapper(self, *args, **kwargs):
            if required_flag_name is not None:
                # Check for explicit kwarg or global flag
                flag_value = kwargs.pop(required_flag_name, globals().get(required_flag_name, False))
                if flag_value is not True:
                    raise RuntimeError(
                        f"{feature_name} is a beta feature. "
                        f"To use it, pass `{required_flag_name}=True` to the constructor "
                        f"or set `{required_flag_name} = True` at the module level."
                    )
                warnings.warn(
                    f"{feature_name} is a beta feature and may change in future releases.",
                    UserWarning,
                    stacklevel=2,
                )
            return init(self, *args, **kwargs)
        return wrapper
    return decorator