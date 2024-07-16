import os
from matplotlib import pyplot as plt


energy_levels_data = {
    1420.20e6: {"name": "1420MHz_HI", "levels": [0, 1, 2, 3]},
    407.8e6: {"name": "408MHz_Haslam", "levels": [0, 1, 2]},
    150.8e6: {"name": "151MHz_6C", "levels": [0, 1, 2]},
    30.0e6: {"name": "50MHz_8C", "levels": [0, 1]},
    322.8e6: {"name": "323MHz_Deuterium", "levels": [0, 1]},
    1610.6e6: {"name": "1611MHz_OH", "levels": [0, 1, 2]},
    1665.2e6: {"name": "1665MHz_OH", "levels": [0, 1, 2]},
    1667.2e6: {"name": "1667MHz_OH", "levels": [0, 1, 2]},
    1720.2e6: {"name": "1720MHz_OH", "levels": [0, 1, 2]},
    2290.8e6: {"name": "2291MHz_H2CO", "levels": [0, 1, 2]},
    2670.8e6: {"name": "2671MHz_RRL", "levels": [0, 1, 2]},
    3260.8e6: {"name": "3261MHz_CH", "levels": [0, 1, 2]},
    3335.8e6: {"name": "3336MHz_CH", "levels": [0, 1, 2]},
    3349.0e6: {"name": "3349MHz_CH", "levels": [0, 1, 2]},
    4829.4e6: {"name": "4830MHz_H2CO", "levels": [0, 1, 2]},
    5289.6e6: {"name": "5290MHz_OH", "levels": [0, 1, 2]},
    5885.0e6: {"name": "5885MHz_CH3OH", "levels": [0, 1, 2]},
    1427.0e6: {"name": "1427MHz_HI", "levels": [0, 1, 2, 3]},
    550.0e6: {"name": "575MHz_HCN", "levels": [0, 1, 2]},
    5500.0e6: {"name": "5550MHz_H2O", "levels": [0, 1, 2]}
}


def create_energy_level_diagram(center_frequency, output_dir, date, time):
    if center_frequency in energy_levels_data:
        molecule_data = energy_levels_data[center_frequency]
        molecule_name = molecule_data["name"]
        energy_levels = molecule_data["levels"]
    else:
        molecule_name = "Unknown"
        energy_levels = [0, 1]  # Example generic energy levels for unknown molecules

    num_levels = len(energy_levels)

    plt.figure(figsize=(8, 6))
    for level in range(num_levels):
        plt.plot([0, 1], [energy_levels[level], energy_levels[level]], marker='o', linestyle='-', linewidth=2, label=f'Level {energy_levels[level]}')

    plt.xlim(0, 1)
    plt.ylim(min(energy_levels) - 0.5, max(energy_levels) + 0.5)
    plt.xlabel('Energy State')
    plt.yticks(energy_levels, [f'E{level}' for level in energy_levels])
    plt.title(f'Energy Level Diagram of {molecule_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, f'energy_level_diagram_{molecule_name}_{date}_{time}.png'))
    
    plt.show()