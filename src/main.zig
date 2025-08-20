// Own implementation of a neural network based and slides of my university course "Nerual Networks and Deepl Learning".
// Only for leanrning purposes (future Exam).
// Currently only on CPU. Not plans for implenting GPU calculations.

const std = @import("std");

const ActivationFunction = struct {
    f: *const fn (f64) f64,
    df: *const fn (f64) f64,
};

const Layer = struct {
    input_size: usize,
    output_size: usize,
    inputs: []f64,
    output: []f64,

    weights: []f64, // weights table
    biases: []f64,

    pre_output: []f64,
    delta_error: []f64,

    activation_fn: ActivationFunction,

    alloc: *std.mem.Allocator,

    pub fn init(alloc: *std.mem.Allocator, input_size: usize, output_size: usize, activation_fn: ActivationFunction) !Layer {
        return .{
            .input_size = input_size,
            .output_size = output_size,
            .inputs = try alloc.alloc(f64, input_size),
            .output = try alloc.alloc(f64, output_size),
            .weights = try alloc.alloc(f64, input_size * output_size),
            .pre_output = try alloc.alloc(f64, output_size),
            .delta_error = try alloc.alloc(f64, output_size),
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
        self.alloc.free(self.pre_output);
        self.alloc.free(self.delta_error);
    }

    fn set_input(self: *Layer, input: []const f64) void {
        std.debug.assert(self.inputs.len == input.len);
        @memcpy(self.inputs, input);
    }

    fn rand_weights(self: *Layer, rng: std.Random) void {
        for (self.weights) |*w| w.* = rng.float(f64) * 0.2 - 0.1;
        for (self.biases) |*b| b.* = 0.0;
    }

    fn compute(self: *Layer) []f64 {
        for (0..self.output_size) |j| {
            var sum: f64 = self.biases[j];
            for (0..self.input_size) |i| {
                sum += self.inputs[i] * self.weights[i * self.output_size + j];
            }
            self.pre_output[j] = sum;
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

    ground_truth: []f64,

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
            .ground_truth = try alloc.alloc(f64, output_size),
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

    fn rand_weights(self: *MLP) void {
        var prng = std.Random.Xoshiro256.init(2893457);
        const rand = prng.random();
        for (self.layers) |*layer| {
            layer.rand_weights(rand);
        }
    }

    fn compute(self: *MLP, inputs: []const f64) []f64 {
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

    fn backprog(self: *MLP, eta: f64) void {
        // const output_delta = self.alloc.alloc(f64, self.output_size);
        const last_layer = &self.layers[self.layers.len - 1];
        for (0..self.output_size) |j| {
            last_layer.delta_error[j] = (self.output[j] - self.ground_truth[j]) * last_layer.activation_fn.df(last_layer.pre_output[j]);
        }

        // Output already calculatet;
        // Input doesnt need to be affected;
        var k: usize = self.layers.len - 2; // start at second-to-last
        while (true) : (k -= 1) {
            const current = &self.layers[k];
            const next = &self.layers[k + 1];

            for (0..current.output_size) |current_idx| {
                var sum: f64 = 0;
                for (0..next.output_size) |next_idx| {
                    sum += next.delta_error[next_idx] * next.weights[current_idx * next.output_size + next_idx];
                }
                current.delta_error[current_idx] =
                    sum * current.activation_fn.df(current.pre_output[current_idx]);
            }

            if (k == 0) break;
        }

        for (self.layers) |*layer| {
            for (0..layer.output_size) |j| {
                for (0..layer.input_size) |i| {
                    const grad = layer.inputs[i] * layer.delta_error[j];
                    layer.weights[i * layer.output_size + j] -= eta * grad;
                }
                layer.biases[j] -= eta * layer.delta_error[j];
            }
        }
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
fn d_sigmoid(t: f64) f64 {
    const s = sigmoid(t);
    return s * (1 - s);
}

const SIGMOID: ActivationFunction = .{
    .f = sigmoid,
    .df = d_sigmoid,
};

fn linear(x: f64) f64 {
    return x;
}

fn d_linear(_: f64) f64 {
    return 1.0;
}

const LINEAR: ActivationFunction = .{
    .f = linear,
    .df = d_linear,
};

fn mse(pred: []const f64, target: []const f64) f64 {
    var sum: f64 = 0;
    for (pred, 0..) |p, i| {
        const diff = p - target[i];
        sum += diff * diff;
    }
    return sum / @as(f64, @floatFromInt(pred.len));
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var alloc = gpa.allocator();

    var hidden_layers = [_]usize{2};
    var activations_fns = [_]ActivationFunction{ SIGMOID, SIGMOID };

    // Architecture: 2 inputs → [2 hidden] → 1 output
    var mlp = try MLP.init(&alloc, 2, 1, hidden_layers[0..], activations_fns[0..]);
    defer mlp.deinit();

    mlp.rand_weights();

    // XOR dataset
    const inputs = [_][2]f64{
        .{ 0, 0 },
        .{ 0, 1 },
        .{ 1, 0 },
        .{ 1, 1 },
    };
    const targets = [_][1]f64{
        .{0},
        .{1},
        .{1},
        .{0},
    };

    const epochs = 100_000;
    const eta = 0.2; // learning rate

    // Train
    for (0..epochs) |epoch| {
        var epoch_loss: f64 = 0;

        for (inputs, 0..) |inp, idx| {
            const out = mlp.compute(inp[0..]); // forward pass
            @memcpy(mlp.ground_truth, &targets[idx]);
            mlp.backprog(eta);
            epoch_loss += mse(out, &targets[idx]);
        }

        if (epoch % 1000 == 0) {
            std.debug.print("Epoch {d}, loss={d:.5}\n", .{ epoch, epoch_loss });
        }
    }

    std.debug.print("\n--- Testing ---\n", .{});
    for (inputs, 0..) |inp, idx| {
        const out = mlp.compute(inp[0..]);
        const out_bin: u8 = if (out[0] > 0.95) 1 else 0;
        std.debug.print("in={any} -> out={d} (raw {d:.3}, target {d})\n", .{ inp, out_bin, out[0], targets[idx][0] });
    }
}
