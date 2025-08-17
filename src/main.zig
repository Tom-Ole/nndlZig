// Own implementation of a neural network based and slides of my university course "Nerual Networks and Deepl Learning".
// Only for leanrning purposes (future Exam).
// Currently only on CPU. Not plans for implenting GPU calculations.

const std = @import("std");

const ActivationFunction = struct {
    f: *const fn (f64) f64,
};

const Layer = struct {
    input_size: usize,
    output_size: usize,
    inputs: []f64,
    output: []f64,

    weights: []f64, // weights table
    biases: []f64,

    activation_fn: ActivationFunction,

    alloc: *std.mem.Allocator,

    pub fn init(alloc: *std.mem.Allocator, input_size: usize, output_size: usize, activation_fn: ActivationFunction) !Layer {
        return .{
            .input_size = input_size,
            .output_size = output_size,
            .inputs = try alloc.alloc(f64, input_size),
            .output = try alloc.alloc(f64, output_size),
            .weights = try alloc.alloc(f64, input_size * output_size),
            .biases = try alloc.alloc(f64, output_size),
            .activation_fn = activation_fn,

            .alloc = alloc,
        };
    }

    pub fn deinit(self: *Layer) void {
        self.alloc.free(self.inputs);
        self.alloc.free(self.weights);
        self.alloc.free(self.biases);
        self.alloc.free(self.output);
    }

    fn set_input(self: *Layer, input: []f64) void {
        std.debug.assert(self.inputs.len == input.len);
        @memcpy(self.inputs, input);
    }

    fn compute(self: *Layer) []f64 {
        for (0..self.output_size) |j| {
            var sum: f64 = self.biases[j];
            for (0..self.input_size) |i| {
                sum += self.inputs[i] * self.weights[i * self.output_size + j];
            }
            self.output[j] = self.activation_fn.f(sum);
        }

        return self.output;
    }

    fn print(self: Layer) void {
        for (self.output) |output| {
            std.debug.print("{d}", .{output});
        }
    }
};

const MLP = struct {
    input_size: usize,
    output_size: usize,
    layer_outputs: []usize,
    layers: []Layer,
    output: []f64,

    alloc: *std.mem.Allocator,

    fn init(alloc: *std.mem.Allocator, input_size: usize, output_size: usize, layer_outputs: []usize, activation_fns: []ActivationFunction) !MLP {
        const layer_count = layer_outputs.len + 1;

        var layers = try alloc.alloc(Layer, layer_count);

        var prev_size = input_size;

        // hidden layers
        for (0..layer_outputs.len) |i| {
            layers[i] = try Layer.init(alloc, prev_size, layer_outputs[i], activation_fns[i]);
            prev_size = layer_outputs[i];
        }

        // output layer
        layers[layer_count - 1] = try Layer.init(alloc, prev_size, output_size, activation_fns[layer_count - 1]);

        return .{
            .alloc = alloc,
            .input_size = input_size,
            .output_size = output_size,
            .layer_outputs = try alloc.dupe(usize, layer_outputs),
            .layers = layers,
            .output = try alloc.alloc(f64, output_size),
        };
    }

    fn deinit(self: *MLP) void {
        self.alloc.free(self.output);
        for (self.layers) |*layer| {
            layer.deinit();
        }
        self.alloc.free(self.layers);
    }

    fn compute(self: *MLP, inputs: []f64) []f64 {
        self.layers[0].set_input(inputs);

        for (self.layers, 0..) |*layer, i| {
            const res = layer.compute();
            if (i + 1 < self.layers.len) {
                self.layers[i + 1].set_input(res);
            }
        }
        @memcpy(self.output, self.layers[self.layers.len - 1].output);
        return self.output;
    }

    fn print(self: MLP) void {
        for (self.output) |output| {
            std.debug.print("{d}", .{output});
        }
    }
};

fn sigmoid(t: f64) f64 {
    return 0.5 * (1 + std.math.tanh(0.5 * t));
}

const SIGMOID: ActivationFunction = .{
    .f = sigmoid,
};

fn linear(x: f64) f64 {
    return x;
}

const LINEAR: ActivationFunction = .{
    .f = linear,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var alloc = gpa.allocator();

    var hidden_layers = [_]usize{2};
    var activations_fns = [_]ActivationFunction{ SIGMOID, SIGMOID };

    // Architecture: 2 inputs → [2 hidden] → 1 output
    var mlp = try MLP.init(&alloc, 2, 1, hidden_layers[0..], activations_fns[0..]);
    defer mlp.deinit();

    var inputs = [_]f64{ 3.0, 1.2 };

    mlp.layers[0].weights[0] = 4.0; // x1 -> h1
    mlp.layers[0].weights[1] = 2.9; // x2 -> h1
    mlp.layers[0].biases[0] = -0.5; // bias of node h1

    mlp.layers[0].weights[2] = -1.0; // x1 -> h2
    mlp.layers[0].weights[3] = 3.1; // x2 -> h2
    mlp.layers[0].biases[1] = 1.11; // bias of h2

    mlp.layers[1].weights[0] = 1.0; // h1 -> out
    mlp.layers[1].weights[1] = 2.0; // h2 -> out
    mlp.layers[1].biases[0] = -0.5;

    // Run forward pass
    _ = mlp.compute(&inputs);
    mlp.print();
}

//
//
//
//  Tests
//
//
//

test "simple Layer with SIGMOID" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var alloc = gpa.allocator();

    var layer: Layer = try Layer.init(
        &alloc,
        2,
        1,
        SIGMOID,
    );
    defer layer.deinit();

    layer.inputs[0] = 1;
    layer.inputs[1] = 0;

    layer.weights[0] = 1;
    layer.weights[1] = 1;

    layer.biases[0] = -1.5;

    // Sigmoid(-0.5)
    const res = layer.compute();
    try std.testing.expectApproxEqAbs(0.3775406687981454, res[0], 1e-9);
}

test "simple MLP with SIGMOID" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var alloc = gpa.allocator();

    var hidden_layers = [_]usize{2};
    var activations_fns = [_]ActivationFunction{ SIGMOID, LINEAR };

    // Architecture: 2 inputs → [2 hidden] → 1 output
    var mlp = try MLP.init(&alloc, 2, 1, hidden_layers[0..], activations_fns[0..]);
    defer mlp.deinit();

    // Example input: (1, 0)
    var inputs = [_]f64{ 1.0, 0.0 };

    // First hidden layer: 2 neurons
    // neuron0 = sigmoid(1*x1 + 1*x2 - 0.5)
    mlp.layers[0].weights[0] = 1.0; // x1 -> h1
    mlp.layers[0].weights[1] = 1.0; // x2 -> h1
    mlp.layers[0].biases[0] = -0.5;

    // neuron1 = sigmoid(1*x1 + 1*x2 - 1.5)
    mlp.layers[0].weights[2] = 1.0; // x1 -> h2
    mlp.layers[0].weights[3] = 1.0; // x2 -> h2
    mlp.layers[0].biases[1] = -1.5;

    // Output layer: 1 neuron
    // sigmoid(h1*1 + h2*1 - 0.5)
    mlp.layers[1].weights[0] = 1.0; // h1 -> out
    mlp.layers[1].weights[1] = 1.0; // h2 -> out
    mlp.layers[1].biases[0] = -0.5;

    // Run forward pass
    _ = mlp.compute(&inputs);
    mlp.print();
}
