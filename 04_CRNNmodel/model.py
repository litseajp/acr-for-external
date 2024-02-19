import torch
import torch.nn as nn
import yaml
from torchsummaryX import summary


class CRNN_model(nn.Module):
    def __init__(self, config):
        super(CRNN_model, self).__init__()
        self.hidden_size = config['hidden_size']
        self.n_chords = config['n_chords']

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1, 3))
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1, 3))
        )

        self.cnn3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 4))
        )

        self.pad = nn.ConstantPad2d((0, 0, 1, 1), 0)

        self.cnn4_1 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=3, padding='valid'),
            nn.BatchNorm2d(80),
            nn.ReLU()
        )

        self.cnn4_2 = nn.Sequential(
            nn.Conv2d(80, 80, kernel_size=3, padding='valid'),
            nn.BatchNorm2d(80),
            nn.ReLU()
        )

        self.gru = nn.GRU(input_size=80, hidden_size=self.hidden_size, num_layers=1,
                          batch_first=True, bidirectional=True)

        self.fc = nn.Linear(self.hidden_size*2, self.n_chords)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.cnn1(x)

        x = self.cnn2(x)

        x = self.cnn3(x)

        x = self.pad(x)
        x = self.cnn4_1(x)

        x = self.pad(x)
        x = self.cnn4_2(x)

        x = x.squeeze(3).permute(0, 2, 1)

        x, _ = self.gru(x, None)

        output = self.fc(x)

        return output


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    with open('CRNN/config.yaml', 'r') as conf:
        config = yaml.safe_load(conf)

    model = CRNN_model(config=config['model']).to(device)
    summary(model, torch.zeros((1, 108, 192)).to(device))
