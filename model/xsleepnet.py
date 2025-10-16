import torch
import torch.nn as nn
# from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

class SleepDataset(Dataset):
    def __init__(self, data, labels, num_epoch, sequence_length):
        self.data = data
        self.labels = labels
        self.num_epoch = num_epoch
        self.sequence_length = sequence_length

        # Remove the tail part that cannot fit into a sequence of sequence_length * num_epoch
        total_length = sequence_length * num_epoch
        extra_length = self.data.shape[2] % total_length
        if extra_length != 0:
            self.data = self.data[:, :, :-extra_length]
            self.labels = self.labels[:, :-extra_length]

        print("Data shape after processing:", self.data.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class XSleepNetFeature(nn.Module):
    def __init__(self, in_channels, num_epoch, dim):
        super(XSleepNetFeature, self).__init__()
        self.num_epoch = num_epoch
        self.dim = dim

        self.conv1 = self.make_layers(in_channels, 16)
        self.conv2 = self.make_layers(16, 16)
        self.conv3 = self.make_layers(16, 32)
        self.conv4 = self.make_layers(32, 32)
        self.conv5 = self.make_layers(32, 64)
        self.conv6 = self.make_layers(64, 64)
        self.conv7 = self.make_layers(64, 128)
        self.conv8 = self.make_layers(128, 128)
        self.conv9 = self.make_layers(128, 256)
        
        self.conv_c5 = nn.Conv1d(256, dim, 1, 1, 0)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(num_epoch)

    def make_layers(self, in_channels, out_channels):
        layer = [nn.Conv1d(in_channels, out_channels, 31, 2, 15)]
        layer.append(nn.BatchNorm1d(out_channels))
        layer.append(nn.PReLU())
        return nn.Sequential(*layer)

    def forward(self, x):
        # Reshape to [batch_size * num_epoch, num_channels, sequence_length]
        batch_size, num_channels, total_length = x.shape
        
        x = x.view(-1, num_channels, total_length)  # Flatten the batch and epoch dimensions

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        # print(x.shape)
        x = self.global_avg_pool(x)
        # print(x.shape)
        x = self.conv_c5(x)
        # print(x.shape)
        return x

if __name__ == '__main__':
    # Example usage
    batch_size = 8
    num_channels = 149
    total_length = 60000
    num_epoch = 180
    sequence_length = total_length // num_epoch
    dim = 5  # Number of scales, corresponding to classes [0, 1, 2, 3, 4]

    data = torch.randn(batch_size, num_channels, total_length)  # [batch_size, num_channels, total_length]
    labels = torch.randint(0, dim, (batch_size, num_epoch))  # Random labels for illustration

    dataset = SleepDataset(data, labels, num_epoch, sequence_length)
    
    # DataLoader takes the dataset and the batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = XSleepNetFeature(in_channels=num_channels, num_epoch=num_epoch, dim=dim)
    for batch_x, batch_y in dataloader:
        print("Batch X shape:", batch_x.shape)  # Expect [batch_size, num_channels, total_length]
        print("Batch Y shape:", batch_y.shape)  # Expect [batch_size, num_epoch]
        output = model(batch_x)
        print("Output shape:", output.shape)  # Expect [batch_size, dim, num_epoch]
        print("Batch Y:", batch_y)
        print("Output:", output)
        break

if __name__ == '__main__':
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the model and move it to the device (GPU or CPU)
    model = XSleepNetFeature(in_channels=149, num_epoch=180, dim=5).to(device)

    # Print the model summary
    print(summary(model, input_size=(149, 60000)))

    # Example input data
    input_data = torch.randn(8, 149, 60000).to(device)  # Move input data to the same device

    # Forward pass (to verify everything works fine)
    output = model(input_data)
    print("Output shape:", output.shape)