import matplotlib.pyplot as plt
import os
import re


def generate_plot(figure_type, file_path, dir_path):
    # Read the episode data from the text file
    with open(file_path, "r") as file:
        text = file.read()

    # Define regular expressions to match episode and total_counter lines
    if figure_type == "#reward":
        pattern = r'eps: (\d+) #reward: tensor\((-?\d+\.\d+)\)'
    elif figure_type == "#total_counter":
        pattern = r'eps: (\d+) #total_counter: (\d+)'
    elif figure_type == "#step":
        pattern = r'eps: (\d+) #step: (\d+)'
    else:
        pattern = r'eps: (\d+) #smart_total_counter: (\d+)'
    print("Pattern:", pattern)

    # Find all episode and total_counter matches
    matches = re.findall(pattern, text)
    # print("Matched?:", matches)

    episodes = []
    item_n = []
    # Extract and print episode number and total_counter
    for episode, item in matches:
        # print(f"Episode: {episode}, Item: {item}")
        episodes.append(int(episode))
        if figure_type == "#reward":
            item_n.append(float(item))
        else:
            item_n.append(int(item))
    # print("item_n for all episode:", item_n)
    # if matches:
    #     final_episode, final_total_counter = matches[-1]

    # Plotting the rewards
    fig = plt.figure(figsize=(8, 5))
    plt.plot(episodes, item_n)
    plt.xlabel('Episode')
    plt.ylabel('{}'.format(figure_type))
    plt.title('{} over Episodes'.format(dir_path))
    plt.show()

    fig.savefig(dir_path + '/{}.png'.format(figure_type))

    file.close()


run_directory = os.getcwd() + '/../runs/'
# List all subdirectories under /runs
sub_dirs = [d for d in os.listdir(run_directory) if os.path.isdir(os.path.join(run_directory, d))]

# Iterating through each subdirectory under /runs
for sub_dir in sub_dirs:
    current_dir = os.path.join(run_directory, sub_dir)
    for dir_path, _, file_names in os.walk(current_dir):
        for file_name in file_names:
            if file_name.endswith('.txt'):
                file_path = os.path.join(dir_path, file_name)

                # Generate a plot based on the data
                if file_name == "num_steps.txt":
                    generate_plot("#reward", file_path, dir_path)
                    generate_plot("#step", file_path, dir_path)
                elif file_name == "total_counter.txt":
                    generate_plot("#total_counter", file_path, dir_path)
                else:
                    generate_plot("#smart_total_counter", file_path, dir_path)
