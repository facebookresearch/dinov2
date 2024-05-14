import os 
import shutil
from pathlib import Path

path = Path('/home/manon/classification/data/Single_cells/vexas')
path_out = Path(os.getcwd()) / 'dataset_bis'

def main():
    for d in os.listdir(path):
        if os.path.isdir(path_out / d):
            os.rmdir(path_out /d)

        os.mkdir(path_out / d)
        files = [f for f in os.listdir(path / d) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
        for f in files[:64]:
            shutil.copy(path / d / f, path_out / d / f)

if __name__ == '__main__':
    main()
