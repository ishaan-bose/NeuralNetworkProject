import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Define the neural network
class ChessEvalNet(nn.Module):
    def __init__(self):
        super(ChessEvalNet, self).__init__()
        self.fc1 = nn.Linear(740, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 1)
        
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = torch.tanh(self.fc5(x))
        x = self.fc6(x)  # No activation on output
        return x

def main():
    print("Loading weights and biases...")
    
    # Create model
    model = ChessEvalNet()
    
    # Load weights
    weights_740 = pd.read_csv('TrainingWeights/weights740.csv', header=None).values  # 512x740
    weights_512 = pd.read_csv('TrainingWeights/weights512.csv', header=None).values  # 256x512
    weights_256 = pd.read_csv('TrainingWeights/weights256.csv', header=None).values  # 128x256
    weights_128 = pd.read_csv('TrainingWeights/weights128.csv', header=None).values  # 64x128
    weights_64 = pd.read_csv('TrainingWeights/weights64.csv', header=None).values    # 16x64
    weights_16 = pd.read_csv('TrainingWeights/weights16.csv', header=None).values    # 1x16
    
    # Load biases (remember: biases740 has 512 values, biases512 has 256, etc.)
    biases_740 = pd.read_csv('TrainingWeights/biases740.csv', header=None).values[0]  # 512 values
    biases_512 = pd.read_csv('TrainingWeights/biases512.csv', header=None).values[0]  # 256 values
    biases_256 = pd.read_csv('TrainingWeights/biases256.csv', header=None).values[0]  # 128 values
    biases_128 = pd.read_csv('TrainingWeights/biases128.csv', header=None).values[0]  # 64 values
    biases_64 = pd.read_csv('TrainingWeights/biases64.csv', header=None).values[0]    # 16 values
    biases_16 = pd.read_csv('TrainingWeights/biases16.csv', header=None).values[0]    # 1 value
    
    # Set weights and biases to model
    model.fc1.weight.data = torch.FloatTensor(weights_740)
    model.fc1.bias.data = torch.FloatTensor(biases_740)
    model.fc2.weight.data = torch.FloatTensor(weights_512)
    model.fc2.bias.data = torch.FloatTensor(biases_512)
    model.fc3.weight.data = torch.FloatTensor(weights_256)
    model.fc3.bias.data = torch.FloatTensor(biases_256)
    model.fc4.weight.data = torch.FloatTensor(weights_128)
    model.fc4.bias.data = torch.FloatTensor(biases_128)
    model.fc5.weight.data = torch.FloatTensor(weights_64)
    model.fc5.bias.data = torch.FloatTensor(biases_64)
    model.fc6.weight.data = torch.FloatTensor(weights_16)
    model.fc6.bias.data = torch.FloatTensor(biases_16)
    
    print("Weights and biases loaded!")
    
    # Create input: all 1's vector
    input_data = torch.ones(1, 740)  # batch size 1, 740 inputs
    target = torch.tensor([[0.6942067]])  # ground truth
    
    print(f"\nInput: vector of all 1's (shape: {input_data.shape})")
    print(f"Target: {target.item()}")
    
    # Forward pass with activation tracking
    print("\nRunning forward pass...")
    model.train()
    
    # Track activations manually
    activations = {}
    activations['input'] = input_data.detach().numpy()[0]
    
    x = input_data
    x = torch.tanh(model.fc1(x))
    activations['layer1'] = x.detach().numpy()[0]
    
    x = torch.tanh(model.fc2(x))
    activations['layer2'] = x.detach().numpy()[0]
    
    x = torch.tanh(model.fc3(x))
    activations['layer3'] = x.detach().numpy()[0]
    
    x = torch.tanh(model.fc4(x))
    activations['layer4'] = x.detach().numpy()[0]
    
    x = torch.tanh(model.fc5(x))
    activations['layer5'] = x.detach().numpy()[0]
    
    output = model.fc6(x)
    activations['output'] = output.detach().numpy()[0]
    
    print(f"Predicted output: {output.item():.10f}")
    
    # Calculate loss (half MSE)
    loss = 0.5 * ((output - target) ** 2)
    print(f"Loss (0.5 * MSE): {loss.item():.10f}")
    
    # Backward pass
    print("\nRunning backward pass...")
    model.zero_grad()
    loss.backward()
    
    print("Gradients computed!")
    
    # Save activations to separate files
    print("\nSaving activations to files...")
    np.savetxt('TestComparison/activations_input.txt', activations['input'], fmt='%.10f')
    np.savetxt('TestComparison/activations_layer1_512.txt', activations['layer1'], fmt='%.10f')
    np.savetxt('TestComparison/activations_layer2_256.txt', activations['layer2'], fmt='%.10f')
    np.savetxt('TestComparison/activations_layer3_128.txt', activations['layer3'], fmt='%.10f')
    np.savetxt('TestComparison/activations_layer4_64.txt', activations['layer4'], fmt='%.10f')
    np.savetxt('TestComparison/activations_layer5_16.txt', activations['layer5'], fmt='%.10f')
    np.savetxt('TestComparison/activations_output.txt', activations['output'], fmt='%.10f')
    
    # Save weight gradients (in row-major order, same as weight files)
    print("Saving weight gradients to files...")
    np.savetxt('TestComparison/gradient_weights740.csv', model.fc1.weight.grad.numpy(), delimiter=',', fmt='%.10f')
    np.savetxt('TestComparison/gradient_weights512.csv', model.fc2.weight.grad.numpy(), delimiter=',', fmt='%.10f')
    np.savetxt('TestComparison/gradient_weights256.csv', model.fc3.weight.grad.numpy(), delimiter=',', fmt='%.10f')
    np.savetxt('TestComparison/gradient_weights128.csv', model.fc4.weight.grad.numpy(), delimiter=',', fmt='%.10f')
    np.savetxt('TestComparison/gradient_weights64.csv', model.fc5.weight.grad.numpy(), delimiter=',', fmt='%.10f')
    np.savetxt('TestComparison/gradient_weights16.csv', model.fc6.weight.grad.numpy(), delimiter=',', fmt='%.10f')
    
    # Save bias gradients (as row vectors to match bias file format)
    print("Saving bias gradients to files...")
    np.savetxt('TestComparison/gradient_biases740.csv', model.fc1.bias.grad.numpy().reshape(1, -1), delimiter=',', fmt='%.10f')
    np.savetxt('TestComparison/gradient_biases512.csv', model.fc2.bias.grad.numpy().reshape(1, -1), delimiter=',', fmt='%.10f')
    np.savetxt('TestComparison/gradient_biases256.csv', model.fc3.bias.grad.numpy().reshape(1, -1), delimiter=',', fmt='%.10f')
    np.savetxt('TestComparison/gradient_biases128.csv', model.fc4.bias.grad.numpy().reshape(1, -1), delimiter=',', fmt='%.10f')
    np.savetxt('TestComparison/gradient_biases64.csv', model.fc5.bias.grad.numpy().reshape(1, -1), delimiter=',', fmt='%.10f')
    np.savetxt('TestComparison/gradient_biases16.csv', model.fc6.bias.grad.numpy().reshape(1, -1), delimiter=',', fmt='%.10f')
    
    print("\nAll files saved!")
    print("\nActivation files:")
    print("  - activations_input.txt (740 values)")
    print("  - activations_layer1_512.txt (512 values)")
    print("  - activations_layer2_256.txt (256 values)")
    print("  - activations_layer3_128.txt (128 values)")
    print("  - activations_layer4_64.txt (64 values)")
    print("  - activations_layer5_16.txt (16 values)")
    print("  - activations_output.txt (1 value)")
    print("\nGradient files:")
    print("  - gradient_weights740.csv through gradient_weights16.csv")
    print("  - gradient_biases740.csv through gradient_biases16.csv")

if __name__ == "__main__":
    main()