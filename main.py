from src.env.environment import CacheDriftEnv

env = CacheDriftEnv()

print("Query 0 in env")
query, _ = env.reset()
print(query)

print("Query 1 in env")
query, reward, terminated, truncated, info = env.step(0)
print(query)