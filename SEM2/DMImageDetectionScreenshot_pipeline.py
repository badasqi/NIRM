import torch
import os
import pandas as pd
import numpy as np
import glob
from PIL import Image
import torchvision.transforms as transforms
from normalization import CenterCropNoPad, get_list_norm
from normalization2 import PaddingWarp
from get_method_here import get_method_here, def_model

def create_csv(data_dir, csv_file):
    # Collect paths to all images in the specified directory and save to a CSV file
    image_paths = glob.glob(os.path.join(data_dir, "*.png")) + glob.glob(os.path.join(data_dir, "*.jpg"))
    df = pd.DataFrame({'src': image_paths})
    df.to_csv(csv_file, index=False)

def run_test(data_path, output_file, weights_dir, csv_file):
    DATA_PATH = data_path
    print("CURRENT OUT FILE")
    print(output_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1

    models_list = {
        'Grag2021_progan': 'Grag2021_progan',
        'Grag2021_latent': 'Grag2021_latent'
    }

    models_dict = {}

    for model_name in models_list:
        _, model_path, arch, _, _ = get_method_here(models_list[model_name], weights_path=weights_dir)
        model = def_model(arch, model_path, localize=False)
        model = model.to(device).eval()
        models_dict[model_name] = model

    with torch.no_grad():
        # Read image paths from the CSV file
        table = pd.read_csv(csv_file)[['src']]
        print("CSV File Content:")
        print(table.head())

        results = []

        for index, row in table.iterrows():
            filename = row['src']
            img = Image.open(filename).convert('RGB')
            img_tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)

            img_results = {'src': filename}

            for model_name, model in models_dict.items():
                output_tensor = model(img_tensor).cpu().numpy()

                if output_tensor.shape[1] == 1:
                    output_tensor = output_tensor[:, 0]
                elif output_tensor.shape[1] == 2:
                    output_tensor = output_tensor[:, 1] - output_tensor[:, 0]
                else:
                    assert False

                if len(output_tensor.shape) > 1:
                    logit = np.mean(output_tensor, (1, 2))
                else:
                    logit = output_tensor[0]

                img_results[model_name] = logit.item()

            results.append(img_results)

    # Write results to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)

def main():
    print("Running the Tests")
    data_dir = "./TestSet/stable_diffusion_outpainting"
    csv_file = os.path.join(data_dir, "operations.csv")
    output_file = "./results_tst/stable_diffusion_outpainting_results.csv"
    weights_dir = "./weights"

    # Create CSV file with image paths
    create_csv(data_dir, csv_file)

    # Run test and save results to a single CSV file
    run_test(data_dir, output_file, weights_dir, csv_file)

if __name__ == "__main__":
    main()