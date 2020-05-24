from pathlib import Path
import shutil

p = Path('.')
to_fetch = 'penalty_*'
output_directory = p / 'final_figures'
output_directory.mkdir(exist_ok=True)
to_explore = p.glob(to_fetch)

for results_dir in to_explore:
    penalty = results_dir.name
    figure_path = p / results_dir / 'figs' / 'final_Q-Values.pdf'
    output_path = output_directory / (penalty + '_final_Q-Values.pdf')
    if figure_path.exists():
        shutil.copyfile(str(figure_path), str(output_path))