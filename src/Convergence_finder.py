import math

def find_min_convergence_region(filename, window_size=10, threshold=20):
    # Read the file and extract eps and values
    with open(filename, 'r') as f:
        lines = f.readlines()

    episodes = []
    values = []
    for line in lines:
        parts = line.split()
        eps = int(parts[1])
        val = int(parts[3])
        episodes.append(eps)
        values.append(val)

    # Search for all convergence regions
    regions = []
    for i in range(len(values) - window_size):
        window_values = values[i:i + window_size]
        if (std := (sum(
                (x - sum(window_values) / window_size) ** 2 for x in window_values) / window_size) ** 0.5) <= threshold:
            regions.append((episodes[i], episodes[i + window_size - 1], sum(window_values) / window_size))

    # If no regions found, return average of last 'window_size' values
    if not regions:
        last_values = values[-window_size:]
        return (episodes[-window_size], episodes[-1], sum(last_values) / window_size)

    # Find the region with the minimum average value
    min_region = min(regions, key=lambda x: x[2])
    return min_region


# Call the function and print results
result = find_min_convergence_region("E:\Summer Research 2023\MADDPG_New\MADDPG\\runs\\1004_223900\BO_TO_MADDPG\config_0330\smart_total_counter.txt")
if result:
    print(
        f"Minimum converged region between episodes {result[0]} and {result[1]} with an average value of {result[2]:.2f}")
else:
    print("No clear convergence region found.")