import argparse
import os
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
from train import Net
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import Dataset

class CustomImageInferenceDataset(Dataset):
    def __init__(self, img_root_dir, transform=None, image_types=()):
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.image_names = []
        for dir, _, files in os.walk(self.img_root_dir):
            for file in files:
                if file.lower().endswith(image_types):
                    self.image_names.append(os.path.join(dir, file)) 

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = self.image_names[idx]
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image

def main():
    # Inference settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Inference")

    parser.add_argument(
        "--model",
        type=str,
        default="mnist_cnn.pt",
        help="trained model to perform inference with",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data_for_inference",
        help="directory containing the data for inference",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA acceleration",
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables MPS acceleration",
    )
    parser.add_argument(
        "--image-types",
        nargs="+",
        default=[".png"],
        help="image types on which the model will perform inference",
    )
    parser.add_argument(
        "--output-file",
        default="inference_output.csv",
        help="the file in which inference results are stored",
    )

    
    args = parser.parse_args()
    loader_kwargs = {"batch_size": args.batch_size}
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    
    if use_cuda:
        loader_kwargs.update({"num_workers": 1, "pin_memory": True, "shuffle": True})
        
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    inference_dataset = CustomImageInferenceDataset(
        img_root_dir=args.target,
        transform=transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        ),
        image_types=tuple(args.image_types)
    )

    inference_loader = torch.utils.data.DataLoader(inference_dataset, **loader_kwargs)

    model = Net().to(device)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()
    
    predictions = torch.tensor([], dtype=torch.int64).to(device)
    for batch_idx, data in enumerate(inference_loader):
        print("Inferencing batch {} of {}".format(batch_idx + 1, len(inference_loader)))
        data = data.to(device)
        output = model(data)
        predictions = torch.cat(
            (
                predictions,
                output.argmax(
                    dim=1, keepdim=True
                )
            )
        )
    pred_counter = Counter(predictions.squeeze().tolist())
    sorted_counts = sorted(pred_counter.items(), key=lambda item: item[1], reverse=True)
    df = pd.DataFrame(sorted_counts, columns=['digit', 'count'])
    if not os.path.exists('results'):
        os.makedirs('results')
    df.to_csv(os.path.join('results', args.output_file), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()