from pathlib import Path
from PIL import Image

inputPath = Path("data/dataset_NB_masked")
inputFiles = inputPath.glob("**/*.png")
outputPath = Path("data/dataset_NB_masked2")

for f in inputFiles:
    print(f.stem)
    outputFile = outputPath / Path(f.stem + ".jpeg")
    im = Image.open(f)
    im.save(outputFile)
