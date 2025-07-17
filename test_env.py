import gym

# This is the crucial line. We are explicitly trying to
# import the module that should contain the environment registration code.
print("Attempting to import libero.envs...")
try:
    import libero.envs
    print("SUCCESS: `libero.envs` imported successfully.")
except ImportError as e:
    print(f"FAILURE: Could not import `libero.envs`. This is likely a missing dependency.")
    print(f"--> ImportError: {e}")
    exit() # Exit the script if the import fails


# Now, we try to create the environment
ENV_NAME = "bridge-v0"
print(f"\nAttempting to create environment: '{ENV_NAME}'...")
try:
    env = gym.make(ENV_NAME)
    print(f"SUCCESS: Environment '{ENV_NAME}' created successfully!")
    print(env)
except gym.error.NameNotFound as e:
    print(f"FAILURE: `gym.make('{ENV_NAME}')` failed.")
    print("This means the environment is not registered under this name, or the registration failed silently.")
    print(f"--> gym.error.NameNotFound: {e}")